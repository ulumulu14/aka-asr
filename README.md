# aka-asr

Polski ASR Conformer-Transducer (RNN-T), trenowany od zera na miksie polskich
korpusów: Mozilla Common Voice PL, Multilingual LibriSpeech (MLS) Polish, BIGOS v2
(Clarin studio/mobile, M-AILABS, FLEURS, korpusy PWR), VoxPopuli PL (Parlament EU)
oraz YODAS PL (manualne napisy z YouTube) — łącznie ~440 h.
Architektura: Conformer encoder + causal-Conformer predictor + joint network,
loss RNN-T (torchaudio) z pomocniczym CTC na encoderze. Opcjonalna augmentacja
waveformu (szum + RIR, OpenSLR 28) — patrz sekcja `augment` w configach.

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

**BIGOS v2, VoxPopuli, YODAS** (pobierane z Hugging Face; audio zapisywane lokalnie do `data/`):

```bash
python utils/datasets_conversion/hf2json.py bigos       # gated: wymaga HF_TOKEN + akceptacji warunków datasetu
python utils/datasets_conversion/hf2json.py voxpopuli
python utils/datasets_conversion/hf2json.py yodas --shards 10   # --shards ogranicza wolumen (~0.9 GB/shard)
```

Manifesty lądują w `data/` jako `{train,val}_manifest_<źródło>.jsonl`.
Manifesty z różnych źródeł można po prostu skonkatenować (`cat a.jsonl b.jsonl > train.jsonl`)
— tak powstają `train_manifest_all.jsonl` / `val_manifest_all.jsonl` używane w `configs/config_*_all.yaml`.
Pełną regenerację całego miksu na maszynie treningowej automatyzuje `runpod/bootstrap.sh`
(patrz `runpod/README.md`).

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

## Tricki treningowe i narzędzia

- **Speed perturbation** (Kaldi 0.9/1.0/1.1×): `augment.speed_perturb: [0.9, 1.0, 1.1]`
  w configu (sample bez czynnika 1.0 omijają feature_cache — wolniejszy DataLoader).
- **Label smoothing aux CTC**: `training.ctc_label_smoothing: 0.05`.
- **Uśrednianie checkpointów** (tanie +0.2–0.5 pp WER, nie do `--resume`):
  ```bash
  python utils/average_checkpoints.py checkpoints/conformer_big/<exp> --last 5
  ```
- **WER greedy vs beam search** (beam + hak shallow-fusion LM w `decoding.py`):
  ```bash
  python utils/evaluate_wer.py --checkpoint <ckpt> --manifest data/val_manifest_all.jsonl --beam-size 4
  ```
- **Pseudo-labeling** (self-training na nieopisanym audio, np. dalsze shardy YODAS):
  ```bash
  python utils/pseudo_label.py --checkpoint <ckpt> --manifest <nieopisane.jsonl> --out data/pseudo.jsonl
  ```
- **ITN** — liczby słowami → cyfry, tylko warstwa wyświetlania (`itn.py`);
  w demo apce włącza ją `ASR_DEMO_ITN=1`.

## Podgląd architektury

```bash
python model_summary.py --size mid
```
