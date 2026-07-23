#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Odpalenie treningu aka-asr na RunPodzie, odlaczone od sesji SSH.
# setsid+nohup -> proces przezyje zamkniecie terminala / restart sesji.
#
#   bash runpod/train.sh                 # swiezy trening
#   RESUME=/workspace/aka-asr/checkpoints/conformer_mid/<exp>/last.pt bash runpod/train.sh
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/aka-asr}"
CONFIG="${CONFIG:-configs/config_runpod_big_bpe.yaml}"
LOG="${LOG:-/workspace/aka-asr/train_runpod.log}"
cd "$REPO_DIR"

# Guard przed podwojnym startem: poprzedni trening (setsid+nohup) zyje w tle,
# a drugi proces na GPU w trybie EXCLUSIVE dostaje "device busy or unavailable"
if pgrep -f "python train.py" >/dev/null; then
  echo "!! Trening juz dziala:"
  pgrep -af "python train.py"
  echo "   Podglad: tail -f $LOG   |   ubicie: pkill -f 'python train.py'"
  exit 1
fi

RESUME_ARG=""
[ -n "${RESUME:-}" ] && RESUME_ARG="--resume $RESUME"

echo "config: $CONFIG"
echo "log:    $LOG"
echo "resume: ${RESUME:-<swiezy trening>}"

setsid nohup python train.py "$CONFIG" $RESUME_ARG > "$LOG" 2>&1 &
echo "PID (wrapper): $!  -- proces pythona zlap: pgrep -f 'python train.py $CONFIG'"

# TensorBoard na porcie 6006 (wystaw port w konfiguracji poda, http proxy RunPod)
if ! pgrep -f "tensorboard.*--logdir" >/dev/null; then
  setsid nohup python -m tensorboard.main \
    --logdir /workspace/aka-asr/checkpoints --host 0.0.0.0 --port 6006 \
    > /workspace/aka-asr/tensorboard.log 2>&1 &
  echo "TensorBoard: port 6006 (wystaw w RunPod -> HTTP proxy)"
fi

echo
echo "Podglad:  tail -f $LOG"
