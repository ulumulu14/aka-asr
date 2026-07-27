#!/usr/bin/env python3
"""
Ewaluacja modelu na held-out tescie: WER + CER, rozbicie per zrodlo, pewnosc,
najgorsze przyklady. Do oceny FINALNEGO modelu na zbiorze NIETKNIETYM w treningu
(nie val -- val steruje wyborem best.pt, wiec zawyza).

Test set: wydziel test splity zrodel (nie uzywane w train/val), np.:
    python utils/datasets_conversion/librispeech2json.py data/mls_polish --split test \
        --out data/test_manifest_mls.jsonl
    python utils/datasets_conversion/cv2json.py cv-*.tar.gz --split test --jsonl \
        --out data/test_manifest_cv.jsonl ...
    cat data/test_manifest_*.jsonl > data/test_manifest_all.jsonl

    python utils/test_model.py --checkpoint checkpoints/conformer_big/<exp>/best.pt \
        --manifest data/test_manifest_all.jsonl [--beam-size 4] [--out report.json]

WER/CER liczone na formie znormalizowanej (lowercase, bez interpunkcji przez
_acoustic) -- porownywalne miedzy runami char/BPE niezaleznie od tego, co model
emituje w interpunkcji/casingu.
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from decoding import beam_search_decode_batch  # noqa: E402
from preprocessing import ASRDataset, make_collate_fn, normalize_text  # noqa: E402
from train import LengthBucketBatchSampler, _edit_distance, compute_wer, greedy_decode_batch  # noqa: E402
from utils.evaluate_wer import load_asr  # noqa: E402

_SOURCES = [("cv-corpus", "cv"), ("mls", "mls"), ("bigos", "bigos"),
            ("voxpopuli", "voxpopuli"), ("yodas", "yodas")]


def _source(path: str) -> str:
    for key, name in _SOURCES:
        if key in path:
            return name
    return "inne"


def _acoustic(text: str) -> str:
    """lowercase + bez interpunkcji -- WER/CER niezalezne od casingu/interpunkcji."""
    return re.sub(r"[^\w\s]", "", normalize_text(text), flags=re.UNICODE)


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    dist = total = 0
    for r, h in zip(refs, hyps):
        dist += _edit_distance(list(r), list(h))
        total += len(r)
    return dist / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out ewaluacja modelu (WER/CER, per-zrodlo)")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--beam-size", type=int, default=0, help="0/1 = greedy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batch-duration", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--worst", type=int, default=10, help="Ile najgorszych przykladow pokazac")
    ap.add_argument("--out", type=Path, default=None, help="Zapisz raport JSON")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer, audio_cfg, max_symbols = load_asr(args.checkpoint, device)

    ds = ASRDataset(
        str(args.manifest), tokenizer, audio_cfg=audio_cfg, augment=False,
        max_duration=20.0, min_duration=0.5, return_index=True,
    )
    if args.limit:
        ds.entries = ds.entries[: args.limit]
        ds.durations = ds.durations[: args.limit]
    sampler = LengthBucketBatchSampler(
        ds.durations, args.batch_size, shuffle=False, max_batch_duration=args.max_batch_duration,
    )
    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=make_collate_fn(blank_id=0),
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    use_beam = args.beam_size and args.beam_size > 1
    mode = f"beam={args.beam_size}" if use_beam else "greedy"
    use_amp = device.type == "cuda"
    rows = []  # (source, ref_norm, hyp_norm, score)
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"test ({mode})", dynamic_ncols=True, ascii=True):
            audio = batch["audio_features"].to(device)
            logit_lengths = batch["logit_lengths"].to(device)
            mask = batch["audio_padding_mask"].to(device)
            idxs = batch["indices"]
            scores = [None] * len(idxs)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                if use_beam:
                    hyp_ids = beam_search_decode_batch(
                        model, audio, logit_lengths, audio_padding_mask=mask,
                        beam_size=args.beam_size, max_symbols_per_step=max_symbols,
                    )
                else:
                    hyp_ids, scores = greedy_decode_batch(
                        model, audio, logit_lengths, audio_padding_mask=mask,
                        max_symbols_per_step=max_symbols, return_scores=True,
                    )
            for i, idx in enumerate(idxs):
                entry = ds.entries[idx]
                ref = _acoustic(entry["text"])
                hyp = _acoustic(tokenizer.decode(hyp_ids[i]))
                rows.append((_source(entry["audio_path"]), ref, hyp, scores[i]))

    # --- metryki ogolne ---
    refs = [r for _, r, _, _ in rows]
    hyps = [h for _, _, h, _ in rows]
    wer = compute_wer(refs, hyps) * 100
    cer = compute_cer(refs, hyps) * 100
    dt = time.time() - t0

    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"test:       {args.manifest}  ({len(rows)} sampli, tryb {mode}, {dt:.0f}s)")
    print(f"\n=== OGOLEM ===\n  WER {wer:.2f}%  |  CER {cer:.2f}%")

    # --- per zrodlo ---
    per = defaultdict(lambda: ([], []))
    for src, r, h, _ in rows:
        per[src][0].append(r)
        per[src][1].append(h)
    print("\n=== PER ZRODLO ===")
    report_src = {}
    for src in sorted(per):
        rr, hh = per[src]
        w = compute_wer(rr, hh) * 100
        c = compute_cer(rr, hh) * 100
        report_src[src] = {"n": len(rr), "wer": round(w, 2), "cer": round(c, 2)}
        print(f"  {src:10} n={len(rr):>6}  WER {w:5.2f}%  CER {c:5.2f}%")

    # --- pewnosc (tylko greedy); pusta hipoteza -> score -inf, licz je osobno ---
    all_scores = [s for _, _, _, s in rows if s is not None]
    n_empty = sum(1 for s in all_scores if s == float("-inf"))
    finite = [s for s in all_scores if s != float("-inf")]
    if all_scores:
        import statistics
        avg = statistics.mean(finite) if finite else float("nan")
        print(f"\n=== PEWNOSC (greedy avg log-prob) ===")
        print(f"  srednia (niepustych): {avg:.3f}  |  pustych hipotez: {n_empty}/{len(all_scores)}")

    # --- najgorsze przyklady ---
    ranked = sorted(rows, key=lambda x: compute_wer([x[1]], [x[2]]), reverse=True)
    print(f"\n=== {args.worst} NAJGORSZYCH (ref -> hyp) ===")
    for src, r, h, _ in ranked[: args.worst]:
        print(f"  [{src}] WER {compute_wer([r],[h]):.0%}\n    REF: {r[:80]!r}\n    HYP: {h[:80]!r}")

    if args.out:
        args.out.write_text(json.dumps({
            "checkpoint": str(args.checkpoint), "manifest": str(args.manifest),
            "mode": mode, "n": len(rows), "wer": round(wer, 2), "cer": round(cer, 2),
            "per_source": report_src,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] raport -> {args.out}")


if __name__ == "__main__":
    main()
