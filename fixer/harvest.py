#!/usr/bin/env python3
"""
Harvest par {bledna hipoteza ASR; poprawna transkrypcja} do treningu fixera.

Uruchamia checkpoint ASR (z train.py) na manifescie, dekoduje greedy DOKLADNIE
jak walidacja (train.greedy_decode_batch) i zapisuje JSONL: {"hyp": ..., "ref": ...}.
Zachowuje rowniez pary hyp==ref -- ucza fixer, zeby NIE psul poprawnych wyjsc.

Wskazowki:
  - pary TRENINGOWE fixera harvestuj z duzego zbioru, najlepiej z --augment
    (SpecAugment przy dekodowaniu -> wiecej bledow = gestsze pary),
  - pary TESTOWE harvestuj z osobnego manifestu BEZ augmentacji (uczciwy WER),
    i NIE trenuj na nich fixera.

    python -m fixer.harvest --checkpoint checkpoints/conformer_big/<exp>/best.pt \
        --manifest data/val_manifest_all.jsonl --out fixer/pairs_val.jsonl \
        --device cuda [--augment] [--max-batch-duration 200] [--limit 20000]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from model import ConformerTransducer  # noqa: E402
from preprocessing import (  # noqa: E402
    ASRDataset,
    AudioConfig,
    CharTokenizer,
    SentencePieceTokenizer,
    make_collate_fn,
    normalize_text,
)
from train import LengthBucketBatchSampler, compute_wer, greedy_decode_batch  # noqa: E402


def load_asr(checkpoint_path: Path, device: torch.device):
    """(model, tokenizer, audio_cfg, max_symbols) z metadanych checkpointu -- jak transcriber."""
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    tok = state["tokenizer"]
    if tok["type"] == "bpe":
        tokenizer = SentencePieceTokenizer(tok["bpe_model_path"])
    else:
        tokenizer = CharTokenizer(alphabet=tok["alphabet"])
    audio_cfg = AudioConfig(**state["audio_cfg"])
    model = ConformerTransducer(
        vocab_size=tokenizer.vocab_size, n_mels=audio_cfg.n_mels, **state["model_cfg"]
    )
    model.load_state_dict(state["model"])
    model.eval().to(device)
    max_symbols = int(state.get("cfg", {}).get("max_symbols_per_step", 3))
    return model, tokenizer, audio_cfg, max_symbols


def main() -> None:
    ap = argparse.ArgumentParser(description="Harvest par {hyp, ref} dla fixera")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--augment",
        action="store_true",
        help="SpecAugment przy dekodowaniu -> wiecej bledow (gestsze pary)",
    )
    ap.add_argument("--max-batch-duration", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="Zatrzymaj po N sample")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument(
        "--lowercase-ref",
        action="store_true",
        help="ref = znormalizowany (lowercase, bez interpunkcji) zamiast oryginalu. "
        "Domyslnie ref zachowuje CASING + interpunkcje -> fixer uczy sie "
        "truecasingu i re-punktuacji, nie tylko korekcji bledow.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer, audio_cfg, max_symbols = load_asr(args.checkpoint, device)

    ds = ASRDataset(
        str(args.manifest),
        tokenizer,
        audio_cfg=audio_cfg,
        augment=args.augment,
        max_duration=20.0,
        min_duration=0.5,
        return_index=True,
    )
    sampler = LengthBucketBatchSampler(
        ds.durations,
        args.batch_size,
        shuffle=False,
        max_batch_duration=args.max_batch_duration,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=make_collate_fn(blank_id=0),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"
    refs_norm: list[str] = []
    hyps_all: list[str] = []
    n = n_diff = n_cased = 0
    t0 = time.time()
    with args.out.open("w", encoding="utf-8") as f, torch.no_grad():
        for bi, batch in enumerate(loader):
            audio = batch["audio_features"].to(device)
            logit_lengths = batch["logit_lengths"].to(device)
            audio_padding_mask = batch["audio_padding_mask"].to(device)
            indices = batch["indices"]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                hyp_ids = greedy_decode_batch(
                    model,
                    audio,
                    logit_lengths,
                    audio_padding_mask=audio_padding_mask,
                    max_symbols_per_step=max_symbols,
                )
            for i in range(audio.size(0)):
                raw = ds.entries[indices[i]]["text_raw"]  # oryginal: casing + interpunkcja
                ref = normalize_text(raw) if args.lowercase_ref else raw
                hyp = tokenizer.decode(hyp_ids[i])  # ASR hyp -- zawsze lowercase
                f.write(json.dumps({"hyp": hyp, "ref": ref}, ensure_ascii=False) + "\n")
                refs_norm.append(normalize_text(raw))
                hyps_all.append(hyp)
                n += 1
                n_diff += hyp != normalize_text(raw)
                n_cased += any(c.isupper() for c in raw)
            if bi % 20 == 0:
                print(
                    f"  {n} par | {n_diff} z bledem | "
                    f"{n / max(1e-9, time.time() - t0):.0f} sampli/s",
                    flush=True,
                )
            if args.limit and n >= args.limit:
                break

    # WER liczony na znormalizowanych (jakosc akustyczna, niezalezna od casingu ref)
    wer = compute_wer(refs_norm, hyps_all) * 100 if refs_norm else float("nan")
    print(f"[OK] {n} par -> {args.out}")
    print(
        f"     z bledem: {n_diff} ({100 * n_diff / max(1, n):.1f}%) | WER hyp vs ref(norm): {wer:.2f}%"
    )
    if not args.lowercase_ref:
        print(
            f"     ref z casingiem: {n_cased}/{n} ({100 * n_cased / max(1, n):.0f}%) sampli "
            f"maja wielkie litery (reszta zrodel lowercase -> ucza korekcji/interpunkcji)"
        )


if __name__ == "__main__":
    main()
