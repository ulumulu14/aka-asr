# aka-asr

Polski ASR Conformer-Transducer (RNN-T), trenowany od zera na połączonych polskojęzycznych
korpusach: Mozilla Common Voice PL, Multilingual LibriSpeech (MLS) Polish, BIGOS v2
(Clarin studio/mobile, M-AILABS, FLEURS, korpusy PWR), VoxPopuli PL (Parlament EU)
oraz YODAS PL - ~440 h audio w miksie bazowym, do ~780 h po rozszerzonym zaciągu
(CV `validated`, YODAS 40 shardów; plan i komendy w `runpod/DATA_SCALING.md`).

Architektura: Conformer encoder + causal-Conformer predictor + joint network,
loss RNN-T (torchaudio) z pomocniczym CTC na encoderze. Augmentacja wave'a
(szum + RIR z OpenSLR 28 + speed perturbation, rolowane NIEZALEŻNIE per sample,
więc wszystkie kombinacje możliwe) plus opcjonalna duplikacja `copies` - patrz
sekcja `augment` w configach. Po dekodowaniu opcjonalny postprocessing
(fixer + ITN, sekcja niżej).

## Instalacja

```bash
pip install -r requirements.txt          # zawiera sentencepiece (tokenizer BPE)
# tylko dla postprocessingu (fixer, plT5) - NIE potrzebne do treningu ASR:
pip install -r requirements-fixer.txt
```

## Przygotowanie danych

Manifesty to JSONL, jeden sample per linia:
`{"audio_path": "...", "duration": 3.45, "text": "transkrypcja"}`

**Common Voice** (wymaga ffmpeg):

```bash
python utils/datasets_conversion/cv2json.py cv-corpus-XX-pl.tar.gz \
    --split validated --jsonl --out cv_manifest.jsonl --workers 8
```

**Multilingual LibriSpeech (MLS) Polish** (flac czytany bezpośrednio, bez konwersji):

```bash
python utils/datasets_conversion/librispeech2json.py /path/to/mls_polish --split train
python utils/datasets_conversion/librispeech2json.py /path/to/mls_polish --split dev
```

**BIGOS v2, VoxPopuli, YODAS** (pobierane z Hugging Face; audio zapisywane lokalnie do `data/`):

```bash
python utils/datasets_conversion/hf2json.py bigos       # gated: wymaga HF_TOKEN + akceptacji warunków datasetu
python utils/datasets_conversion/hf2json.py voxpopuli
python utils/datasets_conversion/hf2json.py yodas --shards 40   # pl000 ma 78 shardów (~1.1 GB / ~9.5 h każdy)
```

Manifesty lądują w `data/` jako `{train,val}_manifest_<źródło>.jsonl`.
Manifesty z różnych źródeł można po prostu skonkatenować (`cat a.jsonl b.jsonl > train.jsonl`)
- tak powstają `train_manifest_all.jsonl` / `val_manifest_all.jsonl` używane w `configs/config_*_all.yaml`.
Odtworzenie miksu datasetów: `runpod/bootstrap.sh`
(patrz `runpod/README.md`).

**Opcjonalnie: cache ficzerów** (mocno przyspiesza DataLoader):

```bash
python features.py train_manifest.jsonl features_cache/ --workers 8
# potem w configu: data.feature_cache_dir: features_cache/
```

**Opcjonalnie: tokenizer BPE** zamiast znakowego:

```bash
python utils/train_bpe.py train_manifest.jsonl --vocab-size 512 --model-prefix bpe_pl
# potem w configu: tokenizer: {type: bpe, bpe_model_path: bpe_pl.model}
```

## Trening

```bash
# lokalnie (CPU / małe GPU):
python train.py configs/config_local.yaml --model_size smol --device cpu

# szybki debug:
python train.py configs/config_local_debug.yaml
```

**RunPod** (dane i checkpointy na network volume w `/workspace`) - trening odpala
`runpod/train.sh` przez `setsid+nohup` (przeżywa rozłączenie SSH) i startuje
TensorBoard na porcie 6006. Config podajesz przez zmienną `CONFIG`:

```bash
cd /workspace/aka-asr
CONFIG=configs/config_runpod_big_bpe.yaml bash runpod/train.sh
tail -f /workspace/aka-asr/train_runpod.log

# wznowienie po ubiciu poda (checkpointy na volume):
RESUME=/workspace/aka-asr/checkpoints/conformer_big/<exp>/last.pt \
  CONFIG=configs/config_runpod_big_bpe.yaml bash runpod/train.sh
```

Pełny runbook (pobranie danych, TensorBoard, skalowanie miksu): `runpod/README.md`
i `runpod/DATA_SCALING.md`.

Rozmiary modelu (`--model_size` albo `model.size` w configu): `smol` (debug),
`mid` (~30M, domyślny), `big` (~131M), `custom` (~319M, do własnych eksperymentów).
`custom` ma domyślne wymiary większe niż `big`, ale przede wszystkim jest do
zdefiniowania przez użytkownika **bez ruszania kodu** - nadpisz dowolne pola w configu:

```yaml
model:
  size: custom
  dims: {enc_dim: 1024, enc_heads: 16, enc_layers: 24, joint_dim: 1024, pred_dim: 1024}
```

`model.dims` działa dla każdego rozmiaru (nadpisuje wybraną bazę); nieznane pole ->
błąd na starcie. Przy `--resume` architektura pochodzi z checkpointu (override ignorowany).

Metryki: `tensorboard --logdir <ckpt_dir>/tb`. Log tekstowy: `<ckpt_dir>/train.log`.

Checkpointy (`best.pt`, `epoch_NNNN.pt`, rolling `last.pt` co `save_every_steps` kroków)
zawierają wagi, stan optymalizatora/schedulera oraz architekturę, tokenizer i config audio -
model da się odtworzyć z samego `.pt`.

Opcje treningu istotne przy większych runach (wszystkie w configu):

- `augment.copies: N` - duplikacja danych: każdy sample N× na epokę, KAŻDA kopia
  z niezależnie losowaną kombinacją augmentacji; łączny budżet widoków = `copies × epochs`.
- `model.blank_bias_init: -4.0` - ujemny bias blanku w jointcie (tylko świeży run);
  przy BPE zapobiega blank-collapse (puste hipotezy, val_WER~100% od epoki 0).
- `training.lr_schedule: cosine` + `min_lr_ratio: 0.1` - warmup -> cosinusowy zjazd
  do floora; na długich runach LR nie zdycha jak przy Noamie (domyślnie: `noam`).
- `training.collapse_abort_epochs: N` - guard: przerwij trening (z zapisem `last.pt`),
  gdy po N epokach val_WER wciąż ~100% - oszczędza godziny GPU na nieudanym starcie.
- `data.bad_list_path: <plik>` - denylist uszkodzonego audio (pliki dekodujące się
  do 0 próbek itp.): wykluczane przy starcie, nowe napotkane w trakcie dopisywane
  cicho (bez spamu w logu); trwa na volume między runami, manifest nietknięty.
- `training.hf_repo: <user>/<repo>` - sync eksperymentu na HuggingFace Hub: metryki
  (TB + log) co epokę, `best.pt` async gdy się poprawia; run = podkatalog
  `conformer_<size>/<timestamp>` w repo. Wymaga `HF_TOKEN` z prawem **zapisu**
  (inny niż read-only do BIGOS); `hf_private: true` domyślnie. Błędy HF nie ubijają
  treningu. Brak `hf_repo` -> sync wyłączony.

## Augmentacja (oprócz noise i RIR), tricki treningowe i narzędzia

- **Speed perturbation** (Kaldi 0.9/1.0/1.1×): `augment.speed_perturb: [0.9, 1.0, 1.1]`
  w configu (sample bez czynnika 1.0 omijają feature_cache - wolniejszy DataLoader).
- **Label smoothing aux CTC**: `training.ctc_label_smoothing: 0.05`.
- **Uśrednianie checkpointów** (tanie +0.2–0.5 pp WER, nie do `--resume`):
  ```bash
  python utils/average_checkpoints.py checkpoints/conformer_big/<exp> --last 5
  ```
- **WER greedy vs beam search** (beam + hak shallow-fusion LM w `decoding.py`):
  ```bash
  python utils/evaluate_wer.py --checkpoint <ckpt> --manifest data/val_manifest_all.jsonl --beam-size 4
  ```
- **Held-out ewaluacja finalnego modelu** (`test.py`): WER + CER, rozbicie
  per źródło (widać gdzie model słaby), pewność, najgorsze przykłady, raport JSON.
  Odpalać na test secie NIETKNIĘTYM w treningu (nie val):
  ```bash
  python test.py --checkpoint <ckpt> --manifest data/test_manifest_all.jsonl --out report.json
  ```
- **Pseudo-labeling / agreement-cleaning** (`utils/pseudo_label.py`): na naprawdę
  nieopisanym audio - pseudo-etykiety z filtrem pewności (`--min-score`); na audio
  z napisami (np. YODAS, napisy manualne) - tryb czyszczenia: dekoduj wszystko
  (`--min-score -99`), potem wyrzuć sample z WER(hyp, napis) > progu.
  Szczegóły i komendy: `runpod/DATA_SCALING.md`.

## Postprocessing (fixer + ITN)`

Po dekodowaniu tekst może przejść przez wspólny pipeline (`postprocessing.PostProcessor`,
ta sama kolejność w demo i w ewaluacji): **AutoFixer** (fine-tune plT5 - korekcja błędów
ASR + truecasing, np. "skarżysko - kamienna"->"Skarżysko-Kamienna", + interpunkcja) -> **ITN** (`itn.py` -
liczby słowami -> cyfry; tylko warstwa wyświetlania, nie stosować przy liczeniu WER,
bo referencje są w formie mówionej) -> **RuleFixer** (`fixer/rules.py` - ręczne poprawki
regex jako ostatnie sito, dla uporczywych błędów których model nie łapie; plik reguł
JSON, wzór w `fixer/rules.example.json`). Przepływ trenowania fixera:
`fixer/harvest.py` (pary {hipoteza, referencja z casingiem} z checkpointu ASR) ->
`fixer/train_fixer.py` (fine-tune) -> `fixer/evaluate.py` (3 osie: WER akustyczny /
interpunkcja F1 / casing recall). W demo włączane przez env: `ASR_DEMO_FIXER=<katalog
modelu>`, `ASR_DEMO_ITN=1`, `ASR_DEMO_RULES=<plik reguł>`. Szczegóły: `fixer/README.md`.

Demo mikrofonowe (Rust + PyO3, nagranie -> checkpoint -> tekst): `aka_asr_demo_app/`.

## Testy

Wąski suite na deterministyczny rdzeń (metryki, tokenizery, normalizacja t0ekstu,
front-end audio, smoke pipeline'u treningowego, round-trip checkpointu) - łapie
ciche bugi kosztujące godziny GPU. Nie testuje zbieżności/WER treningu ani środowiska.

```bash
pip install -r requirements-dev.txt
pytest                      # ~4 s, CPU, bez audio/GPU
```

## Podgląd architektury

```bash
python model_summary.py --size mid
```
