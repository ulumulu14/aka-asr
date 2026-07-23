#!/usr/bin/env python3
"""
WER checkpointu na manifescie -- dekodowanie greedy albo beam search (decoding.py).

    python utils/evaluate_wer.py --checkpoint checkpoints/conformer_big/<exp>/best.pt \
        --manifest data/val_manifest_all.jsonl [--beam-size 4] [--limit 500] \
        [--device cuda] [--max-batch-duration 200]

--beam-size 0/1 = greedy. Typowe uzycie: porownanie greedy vs beam oraz best.pt
vs checkpoint usredniony (utils/average_checkpoints.py). --limit N bierze N
najkrotszych sampli (dataset sortuje po duracji) -- szybkie porownania A/B na
tym samym podzbiorze.

WER liczony na formie mowionej (bez ITN) -- tak jak walidacja treningowa.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from decoding import beam_search_decode_batch  # noqa: E402
from model import ConformerTransducer  # noqa: E402
from preprocessing import (  # noqa: E402
    ASRDataset,
    AudioConfig,
    CharTokenizer,
    SentencePieceTokenizer,
    make_collate_fn,
)
from train import LengthBucketBatchSampler, compute_wer, greedy_decode_batch  # noqa: E402


def load_asr(checkpoint_path: Path, device: torch.device):
    """(model, tokenizer, audio_cfg, max_symbols) z metadanych checkpointu."""
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
    ap = argparse.ArgumentParser(description="WER checkpointu (greedy / beam)")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--beam-size", type=int, default=0, help="0/1 = greedy")
    ap.add_argument("--limit", type=int, default=None, help="N najkrotszych sampli")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batch-duration", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, tokenizer, audio_cfg, max_symbols = load_asr(args.checkpoint, device)

    ds = ASRDataset(
        str(args.manifest),
        tokenizer,
        audio_cfg=audio_cfg,
        augment=False,
        max_duration=20.0,
        min_duration=0.5,
    )
    if args.limit:
        ds.entries = ds.entries[: args.limit]
        ds.durations = ds.durations[: args.limit]
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

    use_beam = args.beam_size and args.beam_size > 1
    mode = f"beam={args.beam_size}" if use_beam else "greedy"
    use_amp = device.type == "cuda"
    refs: list[str] = []
    hyps: list[str] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"dekodowanie ({mode})", dynamic_ncols=True, ascii=True):
            audio = batch["audio_features"].to(device)
            logit_lengths = batch["logit_lengths"].to(device)
            mask = batch["audio_padding_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                if use_beam:
                    hyp_ids = beam_search_decode_batch(
                        model,
                        audio,
                        logit_lengths,
                        audio_padding_mask=mask,
                        beam_size=args.beam_size,
                        max_symbols_per_step=max_symbols,
                    )
                else:
                    hyp_ids = greedy_decode_batch(
                        model,
                        audio,
                        logit_lengths,
                        audio_padding_mask=mask,
                        max_symbols_per_step=max_symbols,
                    )
            targets, target_lengths = batch["targets"], batch["target_lengths"]
            for i in range(audio.size(0)):
                refs.append(tokenizer.decode(targets[i, : int(target_lengths[i])].tolist()))
                hyps.append(tokenizer.decode(hyp_ids[i]))

    wer = compute_wer(refs, hyps) * 100
    dt = time.time() - t0
    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"manifest:   {args.manifest}  ({len(refs)} sampli)")
    print(f"tryb:       {mode}   czas: {dt:.0f}s ({len(refs) / max(1e-9, dt):.1f} sampli/s)")
    print(f"WER:        {wer:.2f}%")
    print("\nprzyklady (ref -> hyp):")
    for r, h in list(zip(refs, hyps))[:3]:
        print(f"  {r!r}\n  -> {h!r}")


if __name__ == "__main__":
    main()
