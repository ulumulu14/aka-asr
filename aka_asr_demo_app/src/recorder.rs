//! Logika nagrywania: cpal → bufor f32 (+ opcjonalnie WAV 16-bit signed PCM).
//!
//! Callback audio robi dwie rzeczy w jednym przebiegu pętli próbek:
//! - konwertuje T → i16 i zapisuje do WAV-a (jeśli writer obecny)
//! - konwertuje T → f32 i pcha do bufora dla Pythona

use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SupportedStreamConfig};
use hound::{WavSpec, WavWriter};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

type WavWriterHandle = Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>;
type AudioBuffer = Arc<Mutex<Vec<f32>>>;

pub struct RecordOptions<'a> {
    pub host: &'a cpal::Host,
    pub device_filter: Option<&'a str>,
    pub output: Option<&'a str>,
    pub max_duration_secs: u64,
}

pub struct RecordResult {
    /// Próbki interleaved [L, R, L, R, ...] w f32
    pub samples: Vec<f32>,
    pub channels: u16,
    pub sample_rate: u32,
}

pub fn list_input_devices(host: &cpal::Host) -> Result<()> {
    println!("Host: {}", host.id().name());
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_else(|| "<brak>".into());
    println!("Domyślne urządzenie wejściowe: {default_name}\n");

    println!("Dostępne urządzenia wejściowe:");
    for (i, dev) in host.input_devices()?.enumerate() {
        let name = dev.name().unwrap_or_else(|_| "<bez nazwy>".into());
        match dev.default_input_config().ok() {
            Some(c) => println!(
                "  [{i:>2}] {name}  ({} Hz, {} kan., {:?})",
                c.sample_rate().0,
                c.channels(),
                c.sample_format()
            ),
            None => println!("  [{i:>2}] {name}  (konfiguracja niedostępna)"),
        }
    }
    Ok(())
}

pub fn record(opts: RecordOptions<'_>) -> Result<RecordResult> {
    // wybór urządzenia
    let device = match opts.device_filter {
        Some(name) => opts
            .host
            .input_devices()?
            .find(|d| d.name().map(|n| n.contains(name)).unwrap_or(false))
            .ok_or_else(|| anyhow!("nie znaleziono urządzenia: {name}"))?,
        None => opts
            .host
            .default_input_device()
            .ok_or_else(|| anyhow!("brak domyślnego urządzenia wejściowego"))?,
    };

    let device_name = device.name().unwrap_or_else(|_| "<bez nazwy>".into());
    let config = device
        .default_input_config()
        .with_context(|| format!("nie udało się pobrać konfiguracji: {device_name}"))?;

    println!("=== ArekSpeechRecognition demo app ===");
    println!("Host:        {}", opts.host.id().name());
    println!("Urządzenie:  {device_name}");
    println!(
        "Konfig:      {} Hz, {} kan., {:?}",
        config.sample_rate().0,
        config.channels(),
        config.sample_format()
    );
    match opts.output {
        Some(p) => println!("Plik WAV:    {p}  (16-bit PCM)"),
        None => println!("Plik WAV:    (pomijam, --no-save)"),
    }
    if opts.max_duration_secs > 0 {
        println!("Limit:       {} s", opts.max_duration_secs);
    } else {
        println!("Limit:       brak (Ctrl+C aby zakończyć)");
    }
    println!("======================");

    // writer WAV (opcjonalny)
    let writer: WavWriterHandle = Arc::new(Mutex::new(if let Some(path) = opts.output {
        let spec = WavSpec {
            channels: config.channels(),
            sample_rate: config.sample_rate().0,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        Some(
            WavWriter::create(path, spec)
                .with_context(|| format!("nie udało się utworzyć: {path}"))?,
        )
    } else {
        None
    }));

    // bufor f32 z preallokacją (proporcjonalnie do limitu albo 60s gdy limit=0)
    let initial_cap = if opts.max_duration_secs > 0 {
        (opts.max_duration_secs as usize)
            * config.sample_rate().0 as usize
            * config.channels() as usize
    } else {
        config.sample_rate().0 as usize * config.channels() as usize * 60
    };
    let buffer: AudioBuffer = Arc::new(Mutex::new(Vec::with_capacity(initial_cap)));
    let counter = Arc::new(AtomicU64::new(0));

    // ctrl-c
    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = stop.clone();
        // ignoruj błąd jeśli handler już zarejestrowany (np. w testach)
        let _ = ctrlc::set_handler(move || {
            stop.store(true, Ordering::SeqCst);
        });
    }

    // strumień cpal
    let stream = build_input_stream(
        &device,
        &config,
        writer.clone(),
        buffer.clone(),
        counter.clone(),
    )?;
    stream.play().context("nie udało się uruchomić strumienia")?;

    println!("\n● Nagrywam... (Ctrl+C aby zakończyć)\n");

    let started = Instant::now();
    let channels_u64 = config.channels() as u64;
    let sr_u64 = config.sample_rate().0 as u64;

    while !stop.load(Ordering::SeqCst) {
        if opts.max_duration_secs > 0 && started.elapsed().as_secs() >= opts.max_duration_secs {
            break;
        }
        let elapsed = started.elapsed().as_secs_f32();
        let frames = counter.load(Ordering::Relaxed) / channels_u64.max(1);
        let secs_recorded = frames as f32 / sr_u64 as f32;
        print!(
            "\r  czas: {:>6.2}s  |  zapisano: {:>6.2}s   ",
            elapsed, secs_recorded
        );
        std::io::stdout().flush().ok();
        std::thread::sleep(Duration::from_millis(200));
    }

    drop(stream); // zatrzymuje wątek audio

    // finalize WAV
    {
        let mut wguard = writer.lock().expect("writer mutex");
        if let Some(w) = wguard.take() {
            w.finalize().context("nie udało się sfinalizować WAV-a")?;
        }
    }

    // przejęcie bufora
    let samples = std::mem::take(&mut *buffer.lock().expect("buffer mutex"));
    let frames = samples.len() / config.channels().max(1) as usize;
    let dur = frames as f32 / sr_u64 as f32;
    println!(
        "\n\n✔ Zarejestrowano {:.2} s ({} próbek/kanał)",
        dur, frames
    );

    Ok(RecordResult {
        samples,
        channels: config.channels(),
        sample_rate: config.sample_rate().0,
    })
}

fn build_input_stream(
    device: &cpal::Device,
    config: &SupportedStreamConfig,
    writer: WavWriterHandle,
    buffer: AudioBuffer,
    counter: Arc<AtomicU64>,
) -> Result<cpal::Stream> {
    let err_fn = |err| eprintln!("\n[!] błąd strumienia audio: {err}");
    let stream_config: cpal::StreamConfig = config.clone().into();

    macro_rules! build {
        ($t:ty) => {{
            let writer = writer.clone();
            let buffer = buffer.clone();
            let counter = counter.clone();
            device.build_input_stream(
                &stream_config,
                move |data: &[$t], _: &_| write_input_data::<$t>(data, &writer, &buffer, &counter),
                err_fn,
                None,
            )?
        }};
    }

    let stream = match config.sample_format() {
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I32 => build!(i32),
        SampleFormat::F32 => build!(f32),
        other => bail!("nieobsługiwany format próbek wejścia: {other:?}"),
    };

    Ok(stream)
}

/// Callback strumienia. Działa w wątku audio — szybkie ścieżki, brak panik.
fn write_input_data<T>(
    input: &[T],
    writer: &WavWriterHandle,
    buffer: &AudioBuffer,
    counter: &AtomicU64,
) where
    T: Sample,
    i16: FromSample<T>,
    f32: FromSample<T>,
{
    // WAV 16-bit
    if let Ok(mut guard) = writer.try_lock() {
        if let Some(w) = guard.as_mut() {
            for &s in input {
                if w.write_sample(i16::from_sample(s)).is_err() {
                    break;
                }
            }
        }
    }

    // bufor f32 dla Pythona
    if let Ok(mut buf) = buffer.try_lock() {
        buf.reserve(input.len());
        for &s in input {
            buf.push(f32::from_sample(s));
        }
    }

    counter.fetch_add(input.len() as u64, Ordering::Relaxed);
}
