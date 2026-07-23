#!/usr/bin/env python3
"""
Usrednianie wag z kilku checkpointow (checkpoint averaging / poor-man's SWA).
Tanie +0.2-0.5 pp WER; sens ma na koncowce treningu (ostatnie K epok).

    python utils/average_checkpoints.py checkpoints/conformer_big/<exp> --last 5
    python utils/average_checkpoints.py e27.pt e28.pt e29.pt --out avg.pt

Wynik niesie pelne metadane (model_cfg/tokenizer/audio_cfg/cfg), wiec dziala
wszedzie tam, gdzie zwykly checkpoint (demo apka, utils/evaluate_wer.py).
NIE nadaje sie do --resume treningu (stan optymalizatora jest wyciety).
"""

import argparse
import re
import sys
from pathlib import Path

import torch


def collect(inputs: list[Path], last: int) -> list[Path]:
    if len(inputs) == 1 and inputs[0].is_dir():
        epochs = sorted(
            inputs[0].glob("epoch_*.pt"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )
        if not epochs:
            sys.exit(f"Brak epoch_*.pt w {inputs[0]}")
        return epochs[-last:]
    return inputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Usrednianie wag checkpointow")
    ap.add_argument("inputs", nargs="+", type=Path, help="Katalog runu ALBO lista plikow .pt")
    ap.add_argument(
        "--last", type=int, default=5, help="Ile ostatnich epok z katalogu (domyslnie 5)"
    )
    ap.add_argument("--out", type=Path, default=None, help="Domyslnie <katalog>/avg_lastK.pt")
    args = ap.parse_args()

    files = collect(args.inputs, args.last)
    if len(files) < 2:
        sys.exit("Potrzeba >= 2 checkpointow do usrednienia")
    print(f"[INFO] usredniam {len(files)} checkpointow:")
    for f in files:
        print(f"   {f}")

    avg: dict | None = None
    base: dict | None = None
    for path in files:
        state = torch.load(path, map_location="cpu", weights_only=False)
        sd = state["model"]
        if avg is None:
            base = state
            avg = {
                k: (v.clone().float() if torch.is_floating_point(v) else v) for k, v in sd.items()
            }
        else:
            if sd.keys() != avg.keys():
                sys.exit(f"{path}: inny zbior kluczy wag niz w {files[0]}")
            for k, v in sd.items():
                if torch.is_floating_point(v):
                    avg[k] += v.float()
                else:
                    avg[k] = v  # tensory calkowite (liczniki BN) -- bierz ostatni

    n = len(files)
    for k, v in base["model"].items():
        if torch.is_floating_point(v):
            avg[k] = (avg[k] / n).to(v.dtype)

    # metadane z pierwszego checkpointu; stan optymalizatora wyciety (nie do resume)
    payload = {k: v for k, v in base.items() if k not in ("optimizer", "scheduler", "scaler")}
    payload["model"] = avg
    payload["averaged_from"] = [f.name for f in files]

    out = args.out or (files[-1].parent / f"avg_last{n}.pt")
    torch.save(payload, out)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
