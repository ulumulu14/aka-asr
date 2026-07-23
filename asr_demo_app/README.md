# asr_demo_app

Konsolowe demo ASR w Rust: nagrywa audio z mikrofonu i transkrybuje je checkpointem
ConformerTransducer wytrenowanym przez `train.py` z tego repo.

**Pipeline:** cpal capture -> bufor f32 -> numpy.ndarray (przez PyO3) -> `transcriber.py`
(mono -> resample 16 kHz -> preemfaza -> log-mel -> CMVN -> greedy transducer decode) -> tekst -> stdout.

Preprocessing i dekodowanie są **identyczne z walidacją treningową** - `transcriber.py`
importuje `model.py`, `preprocessing.py` i `greedy_decode_batch` z `train.py`, a architekturę,
tokenizer i config audio odtwarza z metadanych zapisanych w checkpointcie (`model_cfg`,
`tokenizer`, `audio_cfg`). Żaden config YAML nie jest potrzebny.

## Struktura

```
asr_demo_app/
├── Cargo.toml
├── transcriber.py          # inferencja: checkpoint train.py -> tekst
└── src/
    ├── main.rs             # CLI, auto-wykrywanie checkpointu, orkiestracja
    ├── recorder.rs         # cpal + bufor + WAV
    └── transcribe.rs       # most do Pythona przez PyO3 (sys.path: venv + repo root)
```

## Wymagania

### Build (Rust)

- Toolchain Rust (stable)
- Python ≥ 3.9 z bibliotekami dev - pyo3 0.26 linkuje się z `libpython`
  (repo używa `.venv` z Pythonem 3.14; pyo3 < 0.26 nie wspiera 3.14):
  - **Linux**: `python3-dev` (Debian/Ubuntu) lub `python3-devel` (Fedora)
  - **Windows**: oficjalny installer Pythona
- ALSA dev na Linuxie dla cpal: `libasound2-dev` + `pkg-config`

### Runtime (Python)

Interpreter, którego pyo3 użyje w runtime, musi mieć pakiety repo
(`pip install -r ../requirements.txt` - kluczowe: torch, torchaudio, numpy, tqdm, click, pyyaml).

`transcribe.rs` sam dokleja do `sys.path`:
- site-packages venv - `$VIRTUAL_ENV`, a bez niego `<repo>/.venv`
  (embedded interpreter pyo3 nie aktywuje venv samodzielnie),
- katalog główny repo (importy `model` / `preprocessing` / `train`).

### Wybór interpretera Pythona

pyo3 domyślnie szuka Pythona przez `python3` w PATH. Wersja Pythona z build
musi zgadzać się z runtime (site-packages venv są ładowane do embedded interpretera):

```bash
PYO3_PYTHON=/home/arekk/workspace/aka-asr/.venv/bin/python cargo build --release
```

## Build

```bash
cargo build --release
# binarka: target/release/asr_demo_app
```

## Użycie

Domyślnie (bez `--checkpoint`) apka bierze **najnowszy** `best.pt` - a gdy żadnego
`best.pt` nie ma, najnowszy `last.pt` - z `<repo>/checkpoints/*/*/`.

```bash
cd asr_demo_app

# nagraj (Ctrl+C kończy) i przetranskrybuj najnowszym checkpointem
./target/release/asr_demo_app

# nagraj 10 sekund, konkretny checkpoint
./target/release/asr_demo_app -t 10 -c ../checkpoints/conformer_mid/<exp>/best.pt

# wybierz urządzenie po nazwie, nie zapisuj WAV-a
./target/release/asr_demo_app -d "Scarlett" --no-save

# tylko nagrywanie, bez transkrypcji (stare zachowanie)
./target/release/asr_demo_app --no-transcribe -o nagranie.wav

# transkrypcja istniejącego pliku WAV (bez mikrofonu)
./target/release/asr_demo_app --input-wav test.wav --no-save

# lista urządzeń wejściowych
./target/release/asr_demo_app --list-devices
```

Inferencja idzie domyślnie na **CPU** (żeby nie wchodzić w drogę treningowi na GPU;
mid dekoduje ~5 s audio w ~0.2 s). Wymuszenie GPU:

```bash
ASR_DEMO_DEVICE=cuda ./target/release/asr_demo_app
```

## Test bez mikrofonu

Cały pipeline (Rust -> PyO3 -> checkpoint) można sprawdzić na istniejącym pliku:

```bash
./target/release/asr_demo_app --input-wav test.wav --no-save
```

Sama ścieżka Pythonowa, bez Rusta:

```bash
cd asr_demo_app
../.venv/bin/python -c "
import soundfile as sf, sys; sys.path.insert(0, '.')
import transcriber
data, sr = sf.read('test.wav', dtype='float32', always_2d=True)
print(transcriber.transcribe(data, sr, '../checkpoints/conformer_mid/<exp>/best.pt'))
"
```

## Mikrofon w WSL2

cpal na Linuxie mówi przez ALSA, a w WSL2 nie ma karty dźwiękowej - mikrofon
z Windows jest forwardowany przez serwer PulseAudio WSLg (`PULSE_SERVER` ustawia
system, źródło: `PulseAudioRDPSource`). Żeby ALSA go widziała:

```bash
sudo apt install libasound2-plugins   # moduł pcm_pulse dla ALSA
```

Routing `default` -> Pulse jest już skonfigurowany w `~/.asoundrc`. Po instalacji
`./target/release/asr_demo_app --list-devices` powinno pokazać działający `default`.

## Format danych przekazywanych do Pythona

Funkcja `transcribe(samples, sample_rate, checkpoint_path)` dostaje:

- `samples`: `numpy.ndarray` shape `[frames, channels]`, dtype `float32`, range `[-1.0, 1.0]`
  (cpal `FromSample` skaluje wszystkie formaty wejściowe poprawnie)
- `sample_rate`: `int`, w Hz, natywny dla urządzenia (resampling do 16 kHz robi Python)
- `checkpoint_path`: `str`, ścieżka do checkpointa `.pt` z `train.py`

## Potencjalnie TODO

- **Streaming inference**: chunki w trakcie nagrywania zamiast całego bufora po Ctrl+C.
  Wymaga causal encodera (model wspiera `causal_encoder=True`, obecne checkpointy są non-causal).
- **VAD przed inferencją**: webrtcvad albo silero-vad, żeby ciąć ciszę.
- **Beam search**: obecnie greedy - beam z prefix merge poprawiłby WER o kilka punktów.
- **Serwer**: model jest cache'owany per proces; tryb interaktywny (pętla nagraj->dekoduj)
  amortyzowałby ~1 s ładowania checkpointu.

## Wersje crate'ów

| crate    | wersja  | po co                              |
|----------|---------|------------------------------------|
| cpal     | 0.15    | I/O audio                          |
| hound    | 3.5     | zapis WAV                          |
| pyo3     | 0.26    | embed Python (3.14) + GIL          |
| numpy    | 0.26    | ndarray ↔ numpy                    |
| ndarray  | 0.16    | budowanie [frames, channels]       |
| clap     | 4.5     | CLI                                |
| ctrlc    | 3.4     | obsługa SIGINT                     |
| anyhow   | 1.0     | error handling                     |
