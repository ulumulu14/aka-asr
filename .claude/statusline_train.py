#!/usr/bin/env python3
"""
Statusline Claude Code: pasek postepu treningu ASR.

Parsuje najnowszy checkpoints/conformer_*/*/train.log i renderuje jedna linie:
  mid ▕██████░░░░▏ 61% ep 6/10 krok 10.2k loss 43.1 WER 38.2% ETA 24m
Stany specjalne: walidacja, zakonczony, brak/przerwany trening.
"""

import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/arekk/workspace/aka-asr")
FALLBACK_STEPS_PER_EPOCH = 1700
STALE_S = 480  # brak zapisu w logu > 8 min (walidacja potrafi trwac kilka minut)

sys.stdin.read()  # harness podaje JSON na stdin; nieuzywany


def fmt_k(n: int) -> str:
    return f"{n/1000:.1f}k" if n >= 1000 else str(n)


def main() -> None:
    logs = list(ROOT.glob("checkpoints/conformer_*/*/train.log"))
    log_path = max(logs, key=lambda p: p.stat().st_mtime, default=None)
    if log_path is None:
        print("aka-asr | brak treningu")
        return
    text = log_path.read_text(encoding="utf-8", errors="replace")

    model = re.search(r"\| model: (\w+) \|", text)
    model = model.group(1) if model else "?"

    total_epochs = None
    cfg_m = re.search(r"Config: (\S+) \|", text)
    if cfg_m:
        try:
            cfg_text = (ROOT / cfg_m.group(1)).read_text(encoding="utf-8")
            ep_m = re.search(r"^\s*epochs:\s*(\d+)", cfg_text, re.M)
            total_epochs = int(ep_m.group(1)) if ep_m else None
        except OSError:
            pass

    steps = re.findall(r"\[epoka +(\d+) \| krok +(\d+)\] loss=([\d.]+).*?czas=([\d.]+)m", text)
    vals = re.findall(r"\[epoka +(\d+)\] val_loss=([\d.]+) +val_WER=([\d.]+)%", text)

    if "Koniec treningu" in text:
        best = re.search(r"najlepszy WER=([\d.]+)%", text)
        wer = f" | best WER {best.group(1)}%" if best else ""
        print(f"{model} ✅ trening zakonczony{wer}")
        return

    if not steps:
        print(f"{model} 🔄 start treningu (przygotowanie danych)...")
        return

    epoch, krok, loss, czas_min = (
        int(steps[-1][0]),
        int(steps[-1][1]),
        float(steps[-1][2]),
        float(steps[-1][3]),
    )

    # kroki/epoke: pierwszy krok biezacej epoki / numer epoki (krok jest globalny)
    if epoch > 0:
        first_k = min(int(k) for e, k, _, _ in steps if int(e) == epoch)
        spe = max(1, first_k // epoch)
    else:
        spe = FALLBACK_STEPS_PER_EPOCH

    total = total_epochs or 10
    frac_epoch = min(1.0, max(0.0, (krok - epoch * spe) / spe))
    frac = min(1.0, (epoch + frac_epoch) / total)

    filled = round(frac * 10)
    bar = "▕" + "█" * filled + "░" * (10 - filled) + "▏"

    eta = ""
    if czas_min > 0.5:
        rate = krok / czas_min
        remaining = max(0, total * spe - krok)
        eta_min = remaining / rate
        eta = f" ETA {eta_min/60:.1f}h" if eta_min >= 90 else f" ETA {eta_min:.0f}m"

    wer = f" WER {vals[-1][2]}%" if vals else ""

    age = time.time() - log_path.stat().st_mtime
    tail = text[-400:]
    if "Walidacja..." in tail and age < STALE_S:
        phase = " 🔍 walidacja"
    elif age > STALE_S:
        phase = " ⏸ (log nieaktywny)"
    else:
        phase = ""

    print(
        f"{model} {bar} {frac:.0%} ep {epoch + 1}/{total} "
        f"krok {fmt_k(krok)} loss {loss:.1f}{wer}{eta}{phase}"
    )


if __name__ == "__main__":
    main()
