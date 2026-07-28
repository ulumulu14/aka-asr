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

    python test.py --checkpoint checkpoints/conformer_big/<exp>/best.pt \
        --manifest data/test_manifest_all.jsonl [--beam-size 4] [--out report.json]

BENCHMARK vs WHISPER (uczciwe A/B -- oba na TYCH SAMYCH samplach + TA SAMA
normalizacja, NIE whisper-normalizer):
    python test.py --checkpoint .../best.pt --manifest data/test_manifest_all.jsonl \
        --whisper large-v3
  Wymaga: pip install openai-whisper (osobno, NIE na podzie treningowym) + ffmpeg w PATH.

DWIE METRYKI (--raw-wer przelacza):
  * domyslnie AKUSTYCZNA (_acoustic: lowercase, bez interpunkcji) -- mierzy surowa
    zdolnosc akustyczna, porownywalna char/BPE/Whisper; ITN/casing/interpunkcja
    NIEWIDOCZNE (a ITN wrecz szkodzi -- "dwa"->"2" gdy ref ma "dwa").
  * --raw-wer PRODUKCYJNA (text_raw, casing+interpunkcja zachowane) -- tak jak
    widzi uzytkownik. UWAGA: zrodla z lowercase-ref (voxpopuli/bigos) zawyzaja
    raw-WER (truecasing hyp vs lowercase ref) -- patrz per-zrodlo.

POSTPROCESSING (--postprocess): stosuje PostProcessor.from_env (fixer->ITN->rules)
  do hyp NASZEGO modelu (Whisper ma casing/interpunkcje natywnie, wiec go nie
  ruszamy). Konfiguracja env: ASR_DEMO_FIXER=<dir plT5>, ASR_DEMO_ITN=1,
  ASR_DEMO_RULES=<plik>. Sensowne glownie z --raw-wer (na _acoustic widac tylko
  akustyczna czesc korekcji). Fixer wymaga WYTRENOWANYCH wag (inaczej no-op).
"""
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from decoding import beam_search_decode_batch  # noqa: E402
from postprocessing import PostProcessor  # noqa: E402
from preprocessing import (  # noqa: E402
    AudioConfig,
    CharTokenizer,
    ASRDataset,
    make_collate_fn,
    normalize_text,
)
from train import LengthBucketBatchSampler, _edit_distance, compute_wer, greedy_decode_batch  # noqa: E402
from utils.evaluate_wer import load_asr  # noqa: E402

_SOURCES = [("cv-corpus", "cv"), ("mls", "mls"), ("bigos", "bigos"),
            ("voxpopuli", "voxpopuli"), ("yodas", "yodas")]
# zrodla z lowercase-ref -- raw-WER na nich jest zawyzony (patrz docstring)
_LOWERCASE_REF = {"voxpopuli", "bigos"}


def _source(path: str) -> str:
    for key, name in _SOURCES:
        if key in path:
            return name
    return "inne"


def _acoustic(text: str) -> str:
    """lowercase + bez interpunkcji -- WER/CER niezalezne od casingu/interpunkcji."""
    return re.sub(r"[^\w\s]", "", normalize_text(text), flags=re.UNICODE)


def _raw(text: str) -> str:
    """produkcyjna metryka: zachowaj casing+interpunkcje, normalizuj tylko whitespace."""
    return " ".join(text.split())


def compute_cer(refs: list[str], hyps: list[str]) -> float:
    dist = total = 0
    for r, h in zip(refs, hyps):
        dist += _edit_distance(list(r), list(h))
        total += len(r)
    return dist / max(1, total)


def decode_our_model(model, loader, ds, device, use_beam, beam_size, max_symbols,
                     postproc, ref_field, norm_fn):
    """Dekoduje nasz model po loaderze -> rows (source, ref_norm, hyp_norm, score).
    postproc (PostProcessor|None) stosowany do hyp PRZED normalizacja (produkcja)."""
    use_amp = device.type == "cuda"
    rows = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="nasz model", dynamic_ncols=True, ascii=True):
            audio = batch["audio_features"].to(device)
            logit_lengths = batch["logit_lengths"].to(device)
            mask = batch["audio_padding_mask"].to(device)
            idxs = batch["indices"]
            scores = [None] * len(idxs)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                if use_beam:
                    hyp_ids = beam_search_decode_batch(
                        model, audio, logit_lengths, audio_padding_mask=mask,
                        beam_size=beam_size, max_symbols_per_step=max_symbols,
                    )
                else:
                    hyp_ids, scores = greedy_decode_batch(
                        model, audio, logit_lengths, audio_padding_mask=mask,
                        max_symbols_per_step=max_symbols, return_scores=True,
                    )
            raw_hyps = [ds.tokenizer.decode(hyp_ids[i]) for i in range(len(idxs))]
            if postproc is not None and postproc.enabled:
                raw_hyps = postproc.process_batch(raw_hyps)  # fixer->itn->rules (batch)
            for i, idx in enumerate(idxs):
                entry = ds.entries[idx]
                ref = norm_fn(entry[ref_field])
                hyp = norm_fn(raw_hyps[i])
                rows.append((_source(entry["audio_path"]), ref, hyp, scores[i]))
    return rows


def run_whisper(entries, model_name, language, device_str, ref_field, norm_fn):
    """Transkrypcja Whisperem (openai-whisper) na TYCH SAMYCH entries co nasz model.
    Ref i hyp przechodza przez TE SAMA norm_fn co nasz model -- warunek uczciwego A/B.
    Whisper ma casing/interpunkcje natywnie -> naszego postprocessingu NIE stosujemy.
    Score=None (Whisper nie zwraca naszego log-probu)."""
    try:
        import whisper
    except ImportError:
        sys.exit("Whisper niedostepny: `pip install openai-whisper` (i ffmpeg w PATH)")
    dev = "cuda" if device_str.startswith("cuda") else "cpu"
    print(f"[whisper] laduje model '{model_name}' na {dev} (moze pobierac wagi)...")
    model = whisper.load_model(model_name, device=dev)
    rows = []
    for entry in tqdm(entries, desc=f"whisper {model_name}", dynamic_ncols=True, ascii=True):
        result = model.transcribe(
            entry["audio_path"], language=language, task="transcribe",
            fp16=(dev == "cuda"), verbose=False,
        )
        ref = norm_fn(entry[ref_field])
        hyp = norm_fn(result.get("text", ""))
        rows.append((_source(entry["audio_path"]), ref, hyp, None))
    return rows


def emit_report(rows, label, mode, dt, worst, metric_label):
    """Metryki + wydruk dla jednego zestawu rows. Zwraca dict do JSON/porownania."""
    refs = [r for _, r, _, _ in rows]
    hyps = [h for _, _, h, _ in rows]
    wer = compute_wer(refs, hyps) * 100
    cer = compute_cer(refs, hyps) * 100

    print(f"\n{'='*60}\n{label}")
    print(f"  {len(rows)} sampli, tryb {mode}, metryka {metric_label}, {dt:.0f}s")
    print(f"\n=== OGOLEM ===\n  WER {wer:.2f}%  |  CER {cer:.2f}%")

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
        flag = "  (lowercase-ref: raw-WER zawyzony)" if src in _LOWERCASE_REF and "produkcyjn" in metric_label else ""
        print(f"  {src:10} n={len(rr):>6}  WER {w:5.2f}%  CER {c:5.2f}%{flag}")

    all_scores = [s for _, _, _, s in rows if s is not None]
    if all_scores:
        import statistics
        n_empty = sum(1 for s in all_scores if s == float("-inf"))
        finite = [s for s in all_scores if s != float("-inf")]
        avg = statistics.mean(finite) if finite else float("nan")
        print(f"\n=== PEWNOSC (greedy avg log-prob) ===")
        print(f"  srednia (niepustych): {avg:.3f}  |  pustych hipotez: {n_empty}/{len(all_scores)}")

    ranked = sorted(rows, key=lambda x: compute_wer([x[1]], [x[2]]), reverse=True)
    print(f"\n=== {worst} NAJGORSZYCH (ref -> hyp) ===")
    for src, r, h, _ in ranked[:worst]:
        print(f"  [{src}] WER {compute_wer([r],[h]):.0%}\n    REF: {r[:80]!r}\n    HYP: {h[:80]!r}")

    return {"label": label, "mode": mode, "metric": metric_label, "n": len(rows),
            "wer": round(wer, 2), "cer": round(cer, 2), "per_source": report_src}


def _print_delta(ours, whisper):
    """Porownanie nasz vs Whisper: delta WER/CER ogolem i per zrodlo (minus = my lepsi)."""
    print(f"\n{'='*60}\nPOROWNANIE  (delta = nasz - whisper; UJEMNA = MY LEPSI)")
    dw = ours["wer"] - whisper["wer"]
    dc = ours["cer"] - whisper["cer"]
    print(f"\n=== OGOLEM ===")
    print(f"  nasz    WER {ours['wer']:6.2f}%  CER {ours['cer']:6.2f}%")
    print(f"  whisper WER {whisper['wer']:6.2f}%  CER {whisper['cer']:6.2f}%")
    print(f"  delta   WER {dw:+6.2f}pp CER {dc:+6.2f}pp  -> {'MY LEPSI' if dw < 0 else 'WHISPER LEPSZY'}")
    print("\n=== PER ZRODLO (delta WER) ===")
    for src in sorted(set(ours["per_source"]) | set(whisper["per_source"])):
        o = ours["per_source"].get(src, {}).get("wer")
        w = whisper["per_source"].get(src, {}).get("wer")
        if o is None or w is None:
            continue
        d = o - w
        print(f"  {src:10} nasz {o:6.2f}%  whisper {w:6.2f}%  delta {d:+6.2f}pp"
              f"  {'<-- my' if d < 0 else '<-- whisper'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Held-out ewaluacja modelu (WER/CER, per-zrodlo, vs Whisper)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="nasz model (best.pt)")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--whisper", default=None,
                    help="benchmark: model openai-whisper (large-v3/medium/small/base...)")
    ap.add_argument("--whisper-language", default="pl")
    ap.add_argument("--postprocess", action="store_true",
                    help="stosuj PostProcessor.from_env (fixer->ITN->rules) do hyp naszego modelu")
    ap.add_argument("--raw-wer", action="store_true",
                    help="metryka PRODUKCYJNA (text_raw, casing+interpunkcja) zamiast akustycznej _acoustic")
    ap.add_argument("--beam-size", type=int, default=0, help="0/1 = greedy (dot. naszego modelu)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batch-duration", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--worst", type=int, default=10, help="Ile najgorszych przykladow pokazac")
    ap.add_argument("--out", type=Path, default=None, help="Zapisz raport JSON")
    args = ap.parse_args()

    if not args.checkpoint and not args.whisper:
        ap.error("podaj --checkpoint (nasz model) i/lub --whisper <model>")

    device = torch.device(args.device)

    # metryka: akustyczna (_acoustic na 'text') vs produkcyjna (_raw na 'text_raw')
    ref_field = "text_raw" if args.raw_wer else "text"
    norm_fn = _raw if args.raw_wer else _acoustic
    metric_label = "produkcyjna (casing+interpunkcja)" if args.raw_wer else "akustyczna (_acoustic)"

    # postprocessing (tylko nasz model; Whisper ma casing/interpunkcje natywnie)
    postproc = None
    if args.postprocess:
        postproc = PostProcessor.from_env(device=args.device)
        if not postproc.enabled:
            print("[uwaga] --postprocess podane, ale zaden etap nie wlaczony -- ustaw "
                  "ASR_DEMO_ITN=1 / ASR_DEMO_RULES=<plik> / ASR_DEMO_FIXER=<dir wag plT5>")
        elif not args.raw_wer:
            print("[uwaga] postprocessing na metryce akustycznej: casing/interpunkcja/ITN "
                  "zniknie w _acoustic -- widac tylko korekcje akustyczne. Dodaj --raw-wer.")

    # Tokenizer/audio_cfg z checkpointu (nasz model). Whisper-only: dummy do
    # zbudowania entries -- ASRDataset.__init__ NIE wola tokenizera (poza U>T',
    # tu off), wiec filtr (duration/digits/empty) jest identyczny.
    if args.checkpoint:
        model, tokenizer, audio_cfg, max_symbols = load_asr(args.checkpoint, device)
    else:
        tokenizer, audio_cfg, max_symbols = CharTokenizer(), AudioConfig(), 3

    ds = ASRDataset(
        str(args.manifest), tokenizer, audio_cfg=audio_cfg, augment=False,
        max_duration=20.0, min_duration=0.5, return_index=True,
    )
    if args.limit:
        ds.entries = ds.entries[: args.limit]
        ds.durations = ds.durations[: args.limit]

    reports = {}

    # --- nasz model ---
    if args.checkpoint:
        sampler = LengthBucketBatchSampler(
            ds.durations, args.batch_size, shuffle=False,
            max_batch_duration=args.max_batch_duration,
        )
        loader = DataLoader(
            ds, batch_sampler=sampler, collate_fn=make_collate_fn(blank_id=0),
            num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        )
        use_beam = args.beam_size and args.beam_size > 1
        mode = f"beam={args.beam_size}" if use_beam else "greedy"
        if postproc is not None and postproc.enabled:
            mode += "+postproc"
        t0 = time.time()
        rows = decode_our_model(model, loader, ds, device, use_beam, args.beam_size,
                                max_symbols, postproc, ref_field, norm_fn)
        reports["ours"] = emit_report(
            rows, f"NASZ MODEL ({args.checkpoint.name})", mode, time.time() - t0,
            args.worst, metric_label,
        )

    # --- whisper ---
    if args.whisper:
        t0 = time.time()
        rows_w = run_whisper(ds.entries, args.whisper, args.whisper_language,
                             args.device, ref_field, norm_fn)
        reports["whisper"] = emit_report(
            rows_w, f"WHISPER {args.whisper}", f"whisper-{args.whisper}",
            time.time() - t0, args.worst, metric_label,
        )

    # --- porownanie ---
    if "ours" in reports and "whisper" in reports:
        _print_delta(reports["ours"], reports["whisper"])

    if args.out:
        args.out.write_text(json.dumps(
            {"manifest": str(args.manifest), "metric": metric_label, **reports},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        print(f"\n[OK] raport -> {args.out}")


if __name__ == "__main__":
    main()
