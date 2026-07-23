#!/usr/bin/env python3
"""
Test end-to-end autofixera: WER ASR-samego vs WER ASR+fixer na parach {hyp, ref}.

Uzywa TEJ SAMEJ klasy AutoFixer co inferencja (asr_demo_app/transcriber.py) -> WER
z testu odpowiada temu, co dostaje uzytkownik. Pary bierz z OSOBNEGO manifestu
testowego (harvest.py), NIE z tych, na ktorych trenowano fixer (inaczej przeciek).

    python -m autofixer.evaluate --pairs autofixer/pairs_test.jsonl \
        --fixer checkpoints/autofixer/plt5-base_v1 [--device cuda] [--beams 1] \
        [--guard-max-len-ratio 1.5] [--dump autofixer/changes.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autofixer.fixer import AutoFixer  # noqa: E402
from train import compute_wer  # noqa: E402


def read_pairs(path: Path):
    hyps, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                hyps.append(o.get("hyp", ""))
                refs.append(o.get("ref", ""))
    return hyps, refs


def main() -> None:
    ap = argparse.ArgumentParser(description="Ewaluacja autofixera (WER raw vs fixed)")
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--fixer", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--guard-max-len-ratio", type=float, default=None,
                    help="Odrzuc korekte zmieniajaca dlugosc > tego x (anty-halucynacja)")
    ap.add_argument("--dump", type=Path, default=None, help="Zapisz zmiany (ref/hyp/fixed) do JSONL")
    args = ap.parse_args()

    hyps, refs = read_pairs(args.pairs)
    if not hyps:
        sys.exit(f"Brak par w {args.pairs}")

    fixer = AutoFixer(args.fixer, device=args.device, num_beams=args.beams,
                      guard_max_len_ratio=args.guard_max_len_ratio)

    fixed: list[str] = []
    for i in range(0, len(hyps), args.batch_size):
        fixed.extend(fixer.correct_batch(hyps[i : i + args.batch_size]))
        print(f"  {min(i + args.batch_size, len(hyps))}/{len(hyps)}", end="\r", flush=True)
    print()

    raw = compute_wer(refs, hyps) * 100
    fx = compute_wer(refs, fixed) * 100

    improved = worsened = unchanged = 0
    dump_f = args.dump.open("w", encoding="utf-8") if args.dump else None
    for r, h, fd in zip(refs, hyps, fixed):
        wr_h, wr_f = compute_wer([r], [h]), compute_wer([r], [fd])
        if wr_f < wr_h:
            improved += 1
        elif wr_f > wr_h:
            worsened += 1
        else:
            unchanged += 1
        if dump_f and h != fd:
            dump_f.write(json.dumps({"ref": r, "hyp": h, "fixed": fd}, ensure_ascii=False) + "\n")
    if dump_f:
        dump_f.close()

    print(f"probek:          {len(refs)}")
    print(f"WER ASR-samego : {raw:.2f}%")
    print(f"WER ASR+fixer  : {fx:.2f}%   (delta {fx - raw:+.2f} pp)")
    print(f"na poziomie zdania -> poprawione: {improved} | pogorszone: {worsened} | bez zmian: {unchanged}")
    if worsened > improved:
        print("!! fixer psuje wiecej niz naprawia (over-correction) -- rozwaz --guard-max-len-ratio, "
              "wiecej par albo mniej epok")


if __name__ == "__main__":
    main()
