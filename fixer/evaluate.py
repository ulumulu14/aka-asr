#!/usr/bin/env python3
"""
Test end-to-end fixera: WER ASR-samego vs WER ASR+fixer na parach {hyp, ref}.

Uzywa TEJ SAMEJ klasy AutoFixer co inferencja (asr_demo_app/transcriber.py) -> WER
z testu odpowiada temu, co dostaje uzytkownik. Pary bierz z OSOBNEGO manifestu
testowego (harvest.py), NIE z tych, na ktorych trenowano fixer (inaczej przeciek).

    python -m fixer.evaluate --pairs fixer/pairs_test.jsonl \
        --fixer checkpoints/fixer/plt5-base_v1 [--device cuda] [--beams 1] \
        [--guard-max-len-ratio 1.5] [--dump fixer/changes.jsonl]
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixer.fixer import AutoFixer  # noqa: E402
from preprocessing import normalize_text  # noqa: E402
from train import compute_wer  # noqa: E402

_PUNCT = set(".,!?;:\"'()—–-")


def _acoustic(text: str) -> str:
    """Do WER korekcji: lowercase + bez interpunkcji (izoluje jakosc slow)."""
    return re.sub(r"[^\w\s]", "", normalize_text(text), flags=re.UNICODE)


def _punct_f1(refs: list[str], hyps: list[str]) -> tuple[float, float, float]:
    """F1 znakow interpunkcyjnych (multiset per zdanie -- przyblizenie, ignoruje pozycje)."""
    tp = fp = fn = 0
    for r, h in zip(refs, hyps):
        cr, ch = Counter(c for c in r if c in _PUNCT), Counter(c for c in h if c in _PUNCT)
        for c in set(cr) | set(ch):
            tp += min(cr[c], ch[c])
            fn += max(0, cr[c] - ch[c])
            fp += max(0, ch[c] - cr[c])
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
    return p, rec, f1


def _cap_recall(refs: list[str], hyps: list[str]) -> tuple[float, int]:
    """Recall casingu: ze slow z wielka litera w ref, ile hyp trafil (align po pozycji)."""
    correct = total = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        for i, w in enumerate(rw):
            if w[:1].isalpha() and w[0].isupper():
                total += 1
                if i < len(hw) and hw[i][:1] == w[:1]:
                    correct += 1
    return (correct / total if total else float("nan")), total


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
    ap = argparse.ArgumentParser(description="Ewaluacja fixera (WER raw vs fixed)")
    ap.add_argument("--pairs", required=True, type=Path)
    ap.add_argument("--fixer", required=True, type=Path)
    ap.add_argument("--device", default=None)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--guard-max-len-ratio",
        type=float,
        default=None,
        help="Odrzuc korekte zmieniajaca dlugosc > tego x (anty-halucynacja)",
    )
    ap.add_argument(
        "--dump", type=Path, default=None, help="Zapisz zmiany (ref/hyp/fixed) do JSONL"
    )
    args = ap.parse_args()

    hyps, refs = read_pairs(args.pairs)
    if not hyps:
        sys.exit(f"Brak par w {args.pairs}")

    fixer = AutoFixer(
        args.fixer,
        device=args.device,
        num_beams=args.beams,
        guard_max_len_ratio=args.guard_max_len_ratio,
    )

    fixed: list[str] = []
    for i in range(0, len(hyps), args.batch_size):
        fixed.extend(fixer.correct_batch(hyps[i : i + args.batch_size]))
        print(f"  {min(i + args.batch_size, len(hyps))}/{len(hyps)}", end="\r", flush=True)
    print()

    # 3 osobne osie: korekcja slow (WER akustyczny, bez casingu/interpunkcji),
    # interpunkcja (F1), casing (accuracy). Rozdzielone, bo ref ma teraz casing+
    # interpunkcje a hyp jest lowercase -- laczny "WER" mieszalby te 3 rzeczy.
    wer_raw = compute_wer([_acoustic(r) for r in refs], [_acoustic(h) for h in hyps]) * 100
    wer_fix = compute_wer([_acoustic(r) for r in refs], [_acoustic(h) for h in fixed]) * 100
    p_hyp = _punct_f1(refs, hyps)
    p_fix = _punct_f1(refs, fixed)
    c_hyp, c_tot = _cap_recall(refs, hyps)
    c_fix, _ = _cap_recall(refs, fixed)

    improved = worsened = unchanged = 0
    dump_f = args.dump.open("w", encoding="utf-8") if args.dump else None
    for r, h, fd in zip(refs, hyps, fixed):
        wr_h = compute_wer([_acoustic(r)], [_acoustic(h)])
        wr_f = compute_wer([_acoustic(r)], [_acoustic(fd)])
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

    print(f"probek: {len(refs)}\n")
    print("=== korekcja slow (WER akustyczny, lowercase bez interpunkcji) ===")
    print(f"  ASR-samego : {wer_raw:.2f}%")
    print(f"  ASR+fixer  : {wer_fix:.2f}%   (delta {wer_fix - wer_raw:+.2f} pp)")
    print(
        f"  na poziomie zdania -> poprawione {improved} | pogorszone {worsened} | bez zmian {unchanged}"
    )
    print("\n=== interpunkcja (F1 vs ref; przyblizenie multiset znakow) ===")
    print(f"  ASR-samego : P {p_hyp[0]:.2f} R {p_hyp[1]:.2f} F1 {p_hyp[2]:.2f}")
    print(f"  ASR+fixer  : P {p_fix[0]:.2f} R {p_fix[1]:.2f} F1 {p_fix[2]:.2f}")
    print("\n=== casing (recall: slowa z wielka litera w ref, ktore fixer trafil) ===")
    if c_tot:
        print(f"  ASR-samego : {c_hyp:.2%}   ASR+fixer : {c_fix:.2%}   ({c_tot} slow z casingiem)")
    else:
        print("  (ref bez casingu -- np. voxpopuli/bigos lowercase; zmierz na CV/MLS/YODAS)")
    if worsened > improved:
        print(
            "\n!! fixer psuje wiecej slow niz naprawia (over-correction) -- rozwaz "
            "--guard-max-len-ratio, wiecej par albo mniej epok"
        )
    print(
        "\nWniosek: jesli fixer mocno podnosi interpunkcje-F1 i casing, osobny "
        "re-punctuator/truecaser NIE jest potrzebny (fixer to lapie)."
    )


if __name__ == "__main__":
    main()
