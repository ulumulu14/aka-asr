#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Bootstrap treningu aka-asr na RunPodzie.
#
# Zaklada: pod z GPU (A100/4090) + NETWORK VOLUME zamontowany w /workspace
# (dane i checkpointy MUSZA lezec na volume, zeby przezyly ubicie poda).
#
# Regeneruje pelny miks danych z zewnetrznych zrodel (tempo datacenter):
#   bigos/voxpopuli/yodas -> Hugging Face (utils/datasets_conversion/hf2json.py)
#   mls                   -> OpenSLR/fbaipublicfiles (tar.gz + librispeech2json.py)
#   RIRS_NOISES           -> OpenSLR 28 (augmentacja)
# CV (~5.7 GB) NIE jest regenerowany tutaj (cv2json wymaga surowego korpusu Mozilli).
# Dosylasz go z lokala -- patrz runpod/README.md, sekcja "CV".
#
# Idempotentny: kazdy krok pomija sie, jesli manifest/katalog juz istnieje.
# Uzycie:
#   export HF_TOKEN=hf_...            # token z akceptacja warunkow BIGOS
#   bash runpod/bootstrap.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/aka-asr}"
DATA_DIR="$REPO_DIR/data"
YODAS_SHARDS="${YODAS_SHARDS:-10}"
MLS_URL="${MLS_URL:-https://dl.fbaipublicfiles.com/mls/mls_polish.tar.gz}"
RIRS_URL="${RIRS_URL:-https://www.openslr.org/resources/28/rirs_noises.zip}"

cd "$REPO_DIR"
mkdir -p "$DATA_DIR"
export HF_HUB_ENABLE_HF_TRANSFER=1   # szybsze pobieranie z HF

log() { echo -e "\n\033[1;36m==> $*\033[0m"; }

# --- 1. Zaleznosci ----------------------------------------------------------
log "Instaluje zaleznosci systemowe i pythonowe"
apt-get update -qq && apt-get install -y -qq ffmpeg unzip wget >/dev/null
pip install -q -r requirements.txt
# hf_transfer jawnie -- extra huggingface_hub[hf_transfer] usuniety w hub 1.x
# (pip warningowal, akcelerator sie nie instalowal, a HF_HUB_ENABLE_HF_TRANSFER=1 rzucalby blad)
pip install -q hf_transfer huggingface_hub datasets pyarrow soundfile

if [ -z "${HF_TOKEN:-}" ]; then
  echo "!! HF_TOKEN nie ustawiony -- BIGOS (gated) sie nie pobierze. Ustaw i uruchom ponownie."
fi

# --- 2. Zrodla z HF (bigos / voxpopuli / yodas) -----------------------------
HF2JSON="python utils/datasets_conversion/hf2json.py --out-root $DATA_DIR --manifest-dir $DATA_DIR"

if [ ! -f "$DATA_DIR/train_manifest_yodas.jsonl" ]; then
  log "YODAS ($YODAS_SHARDS shardow)"; $HF2JSON yodas --shards "$YODAS_SHARDS"
else log "YODAS -- manifest istnieje, pomijam"; fi

if [ ! -f "$DATA_DIR/train_manifest_voxpopuli.jsonl" ]; then
  log "VoxPopuli"; $HF2JSON voxpopuli
else log "VoxPopuli -- manifest istnieje, pomijam"; fi

if [ ! -f "$DATA_DIR/train_manifest_bigos.jsonl" ]; then
  log "BIGOS (gated -- wymaga HF_TOKEN)"; $HF2JSON bigos
else log "BIGOS -- manifest istnieje, pomijam"; fi

# --- 3. MLS Polish (OpenSLR) ------------------------------------------------
if [ ! -f "$DATA_DIR/train_manifest_mls.jsonl" ]; then
  log "MLS Polish -- pobieram i rozpakowuje"
  if [ ! -d "$DATA_DIR/mls_polish/train" ]; then
    wget -q --show-progress -O "$DATA_DIR/mls_polish.tar.gz" "$MLS_URL"
    # --no-same-owner: na network volume chown na uid z archiwum failuje (Invalid
    # argument) -> tar konczy exit!=0 -> set -e ubijaloby bootstrap. Pliki i tak sa
    # wypakowane (chown jest po zapisie), wiec ownership pomijamy.
    tar --no-same-owner -xzf "$DATA_DIR/mls_polish.tar.gz" -C "$DATA_DIR"
    rm -f "$DATA_DIR/mls_polish.tar.gz"
  fi
  python utils/datasets_conversion/librispeech2json.py "$DATA_DIR/mls_polish" \
    --split train --out "$DATA_DIR/train_manifest_mls.jsonl"
  python utils/datasets_conversion/librispeech2json.py "$DATA_DIR/mls_polish" \
    --split dev   --out "$DATA_DIR/val_manifest_mls.jsonl"
else log "MLS -- manifest istnieje, pomijam"; fi

# --- 4. RIRS_NOISES (augmentacja) ------------------------------------------
if [ ! -d "$DATA_DIR/RIRS_NOISES" ]; then
  log "RIRS_NOISES (OpenSLR 28)"
  wget -q --show-progress -O "$DATA_DIR/rirs.zip" "$RIRS_URL"
  unzip -q "$DATA_DIR/rirs.zip" -d "$DATA_DIR"
  rm -f "$DATA_DIR/rirs.zip"
else log "RIRS_NOISES -- istnieje, pomijam"; fi

# --- 5. CV (dosylany z lokala) ---------------------------------------------
# Oczekiwane: $DATA_DIR/train_manifest_cv.jsonl i val_manifest_cv.jsonl z audio_path
# wskazujacym na /workspace/aka-asr/data/cv-corpus-.../wav/... (patrz README, remap sed).
if [ ! -f "$DATA_DIR/train_manifest_cv.jsonl" ]; then
  echo "!! Brak $DATA_DIR/train_manifest_cv.jsonl -- CV nie zostal dostarczony."
  echo "   Manifest ALL powstanie BEZ CV. Dosli CV wg runpod/README.md i uruchom ponownie."
fi

# --- 6. Sklejka manifestow (train = wszystko, val = bez yodas) --------------
log "Skladam train/val_manifest_all.jsonl"
cat_if() { for f in "$@"; do [ -f "$f" ] && cat "$f"; done; }

cat_if \
  "$DATA_DIR/train_manifest_cv.jsonl" \
  "$DATA_DIR/train_manifest_mls.jsonl" \
  "$DATA_DIR/train_manifest_bigos.jsonl" \
  "$DATA_DIR/train_manifest_voxpopuli.jsonl" \
  "$DATA_DIR/train_manifest_yodas.jsonl" \
  > "$DATA_DIR/train_manifest_all.jsonl"

cat_if \
  "$DATA_DIR/val_manifest_cv.jsonl" \
  "$DATA_DIR/val_manifest_mls.jsonl" \
  "$DATA_DIR/val_manifest_bigos.jsonl" \
  "$DATA_DIR/val_manifest_voxpopuli.jsonl" \
  > "$DATA_DIR/val_manifest_all.jsonl"

log "Gotowe. Statystyki:"
echo "  train: $(wc -l < "$DATA_DIR/train_manifest_all.jsonl") sampli"
echo "  val:   $(wc -l < "$DATA_DIR/val_manifest_all.jsonl") sampli"
echo
echo "Nastepny krok:  bash runpod/train.sh"
