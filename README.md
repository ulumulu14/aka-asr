# aka-asr

Polski ASR Conformer-Transducer (RNN-T), trenowany od zera na Mozilla Common Voice PL
+ Multilingual LibriSpeech (MLS) Polish.
Architektura: Conformer encoder + causal-Conformer predictor + joint network,
loss RNN-T (torchaudio) z pomocniczym CTC na encoderze.

## Instalacja

```bash
pip install -r requirements.txt
# opcjonalnie, dla tokenizera BPE:
pip install sentencepiece
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

Manifesty z różnych źródeł można po prostu skonkatenować (`cat a.jsonl b.jsonl > train.jsonl`).

**Opcjonalnie: cache cech** (mocno przyspiesza DataLoader):

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

# RunPod (dane na network volume w /workspace):
python train.py configs/config_runpod.yaml

# wznowienie po padzie maszyny:
python train.py configs/config_runpod.yaml --resume /workspace/checkpoints/conformer_mid/<exp>/last.pt
```

Rozmiary modelu: `smol` (debug), `mid` (~30M, domyślny), `big` (`--model_size` albo `model.size` w configu).

Metryki: `tensorboard --logdir <ckpt_dir>/tb`. Log tekstowy: `<ckpt_dir>/train.log`.

Checkpointy (`best.pt`, `epoch_NNNN.pt`, rolling `last.pt` co `save_every_steps` kroków)
zawierają wagi, stan optymalizatora/schedulera oraz architekturę, tokenizer i config audio —
model da się odtworzyć z samego `.pt`.

## Podgląd architektury

```bash
python model_summary.py --size mid
```
