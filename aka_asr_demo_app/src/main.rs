//! Rejestrator dźwięku → preprocess (Python) → inferencja ConformerTransducer → transkrypcja.
//!
//! Pipeline:
//! 1. cpal capture do bufora f32 (i równolegle do WAV 16-bit jeśli włączone)
//! 2. po Ctrl+C / timeout: Vec<f32> → numpy.ndarray przez PyO3
//! 3. transcriber.py:transcribe(samples, sample_rate, checkpoint) → string
//! 4. wypisanie transkrypcji
//!
//! Checkpoint bez --checkpoint znajdowany automatycznie: najnowszy best.pt
//! (a w braku — last.pt) w <repo>/checkpoints/*/*/.
mod recorder;
mod transcribe;

use anyhow::{anyhow, Result};
use clap::Parser;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Rejestrator dźwięku z transkrypcją Conformer-Transducer",
    long_about = "Nagrywa audio z wybranego urządzenia, opcjonalnie zapisuje do WAV (16-bit PCM), \
                  po czym przekazuje bufor do transcriber.py, który liczy log-mel + CMVN \
                  i dekoduje greedy przez checkpoint ConformerTransducer z train.py. \
                  Bez --checkpoint bierze najnowszy best.pt/last.pt z <repo>/checkpoints."
)]
struct Args {
    /// Ścieżka pliku WAV (16-bit PCM)
    #[arg(short, long, default_value = "recording.wav")]
    output: String,

    /// Pomiń zapis WAV-a (tylko transkrypcja)
    #[arg(long)]
    no_save: bool,

    /// Tylko nagrywanie, bez transkrypcji
    #[arg(long)]
    no_transcribe: bool,

    /// Wypisz dostępne urządzenia wejściowe i zakończ
    #[arg(short, long)]
    list_devices: bool,

    /// Fragment nazwy urządzenia wejściowego (case-sensitive)
    #[arg(short, long)]
    device: Option<String>,

    /// Limit czasu nagrywania w sekundach (0 = bez limitu, Ctrl+C kończy)
    #[arg(short = 't', long, default_value_t = 0)]
    duration: u64,

    /// Checkpoint .pt z train.py (domyślnie: najnowszy best.pt/last.pt z <repo>/checkpoints)
    #[arg(short = 'c', long)]
    checkpoint: Option<PathBuf>,

    /// Transkrybuj istniejący plik WAV zamiast nagrywać z mikrofonu
    #[arg(short = 'i', long)]
    input_wav: Option<PathBuf>,

    /// Ścieżka do skryptu Python z funkcją transcribe()
    #[arg(long, default_value = "transcriber.py")]
    python_script: PathBuf,
}

/// Najnowszy checkpoint w układzie checkpoints/<rodzina>/<eksperyment>/{best,last}.pt.
/// best.pt ma pierwszeństwo (wybrany po val_WER); last.pt to fallback z trwającego treningu.
fn find_default_checkpoint(ckpt_root: &Path) -> Option<PathBuf> {
    let mut best: Vec<(SystemTime, PathBuf)> = Vec::new();
    let mut last: Vec<(SystemTime, PathBuf)> = Vec::new();

    for family in std::fs::read_dir(ckpt_root).ok()?.flatten() {
        let Ok(experiments) = std::fs::read_dir(family.path()) else {
            continue;
        };
        for exp in experiments.flatten() {
            for (name, bucket) in [("best.pt", &mut best), ("last.pt", &mut last)] {
                let path = exp.path().join(name);
                if let Ok(mtime) = path.metadata().and_then(|m| m.modified()) {
                    bucket.push((mtime, path));
                }
            }
        }
    }

    for bucket in [&mut best, &mut last] {
        if let Some((_, path)) = bucket.iter().max_by_key(|(t, _)| *t) {
            return Some(path.clone());
        }
    }
    None
}

/// WAV -> interleaved f32 w [-1, 1]; ten sam format co bufor z recorder::record.
fn load_wav(path: &Path) -> Result<recorder::RecordResult> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow!("nie udało się otworzyć {}: {e}", path.display()))?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let denom = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / denom))
                .collect::<Result<_, _>>()?
        }
    };
    Ok(recorder::RecordResult {
        samples,
        channels: spec.channels,
        sample_rate: spec.sample_rate,
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let host = cpal::default_host();

    if args.list_devices {
        return recorder::list_input_devices(&host);
    }

    // ścieżki rozwiązywane przed nagrywaniem — żeby user nie tracił nagrania na typo
    let transcription_plan = if args.no_transcribe {
        None
    } else {
        let script = args.python_script.canonicalize().map_err(|e| {
            anyhow!(
                "skrypt python nie istnieje: {} ({e})",
                args.python_script.display()
            )
        })?;

        let checkpoint = match &args.checkpoint {
            Some(ckpt) => {
                if !ckpt.exists() {
                    return Err(anyhow!("checkpoint nie istnieje: {}", ckpt.display()));
                }
                ckpt.clone()
            }
            None => {
                let ckpt_root = script
                    .parent()
                    .and_then(Path::parent)
                    .map(|repo| repo.join("checkpoints"))
                    .ok_or_else(|| anyhow!("nie można wyprowadzić katalogu repo ze ścieżki skryptu"))?;
                find_default_checkpoint(&ckpt_root).ok_or_else(|| {
                    anyhow!(
                        "nie znaleziono best.pt/last.pt w {} — podaj --checkpoint \
                         (albo --no-transcribe, jeśli chcesz tylko nagrać)",
                        ckpt_root.display()
                    )
                })?
            }
        };
        println!("[i] checkpoint: {}", checkpoint.display());
        Some((checkpoint, script))
    };

    // ---- audio: plik WAV albo nagrywanie ----
    let result = match &args.input_wav {
        Some(path) => {
            println!("[i] wczytuję {}", path.display());
            load_wav(path)?
        }
        None => recorder::record(recorder::RecordOptions {
            host: &host,
            device_filter: args.device.as_deref(),
            output: if args.no_save { None } else { Some(&args.output) },
            max_duration_secs: args.duration,
        })?,
    };

    // ---- transkrypcja ----
    if let Some((checkpoint, script)) = transcription_plan {
        if result.samples.is_empty() {
            return Err(anyhow!("bufor audio jest pusty — nie ma czego transkrybować"));
        }

        println!(
            "\n[i] uruchamiam transkrypcję ({} próbek, {} kanałów, {} Hz)...",
            result.samples.len(),
            result.channels,
            result.sample_rate
        );

        let text = transcribe::run(
            result.samples,
            result.channels,
            result.sample_rate,
            &checkpoint,
            &script,
        )?;

        println!("\n=== Transkrypcja ===\n{}\n====================", text.trim());
    }

    Ok(())
}
