#!/usr/bin/env python3
"""
Trening glowicy V/A na ZAMROZONYM encoderze ASR (affect v1).

    python -m affect.train_head --checkpoint checkpoints/conformer_big/<exp>/best.pt \
        --manifest data/nemo_affect.jsonl --out affect_head.pt [--epochs 30]

Split per MOWCA (speaker-independent; hash speaker_id -> ~20% val) -- split per-sample
zawyzalby CCC (glowica nauczylaby sie glosow, nie emocji). Encoder w eval/no_grad
(bf16 na cuda); trenuje sie tylko glowica (~1M param) -> na RTX 5080 minuty.
Checkpoint glowicy niesie meta: sciezke encodera, mapowanie V/A, metryki val i
FLAGE LICENCJI zrodla danych (nEMO=NC -> prototyp).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from affect.head import AffectHead, ccc, ccc_loss, load_frozen_encoder  # noqa: E402
from preprocessing import AudioConfig, LogMelExtractor, cmvn, subsampled_length  # noqa: E402


def _val_speaker(spk: str, frac: int = 5) -> bool:
    """Deterministycznie: ~1/frac mowcow idzie do val (md5, nie hash() -- ten jest per-proces)."""
    return int(hashlib.md5(spk.encode()).hexdigest(), 16) % frac == 0


def _batches(entries, batch_size):
    order = sorted(range(len(entries)), key=lambda i: entries[i]["duration"])
    for i in range(0, len(order), batch_size):
        yield [entries[j] for j in order[i: i + batch_size]]


def _forward(batch, extractor, encoder, head, device, train_head_mode):
    feats = [cmvn(extractor(e["audio_path"])) for e in batch]
    lens = [f.shape[0] for f in feats]
    T = max(lens)
    x = torch.zeros(len(feats), T, feats[0].shape[1])
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = torch.from_numpy(f)
    sub_lens = [subsampled_length(l) for l in lens]
    Ts = max(sub_lens)
    mask = torch.ones(len(feats), Ts, dtype=torch.bool)
    for i, sl in enumerate(sub_lens):
        mask[i, :sl] = False
    x, mask = x.to(device), mask.to(device)
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=use_amp):
        enc = encoder(x, key_padding_mask=mask)
    pred = head(enc.float()[:, :Ts], mask)
    target = torch.tensor([[e["valence"], e["arousal"]] for e in batch], device=device)
    return pred, target


def main() -> None:
    ap = argparse.ArgumentParser(description="Trening glowicy affect V/A (frozen encoder)")
    ap.add_argument("--checkpoint", required=True, type=Path, help="best.pt ASR (encoder)")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("affect_head.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data-license", default="CC-BY-NC-SA (nEMO) -- PROTOTYP, nie komercyjnie")
    args = ap.parse_args()

    device = torch.device(args.device)
    encoder, enc_dim, audio_cfg, _ = load_frozen_encoder(args.checkpoint, device)
    extractor = LogMelExtractor(AudioConfig(**audio_cfg) if audio_cfg else AudioConfig())
    head = AffectHead(enc_dim).to(device)
    n_par = sum(p.numel() for p in head.parameters())

    entries = [json.loads(l) for l in open(args.manifest, encoding="utf-8")]
    train = [e for e in entries if not _val_speaker(e["speaker_id"])]
    val = [e for e in entries if _val_speaker(e["speaker_id"])]
    spk_t = {e["speaker_id"] for e in train}
    spk_v = {e["speaker_id"] for e in val}
    assert not (spk_t & spk_v)
    print(f"glowica {n_par/1e6:.2f}M | train {len(train)} ({len(spk_t)} mowcow) "
          f"| val {len(val)} ({len(spk_v)} mowcow) | encoder frozen {enc_dim}d")

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    rng = np.random.default_rng(42)
    best = -1.0
    for ep in range(args.epochs):
        head.train()
        rng.shuffle(train)
        t0, losses = time.time(), []
        for batch in _batches(train, args.batch_size):
            pred, target = _forward(batch, extractor, encoder, head, device, True)
            loss = ccc_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        head.eval()
        with torch.no_grad():
            preds, targets = [], []
            for batch in _batches(val, args.batch_size):
                p, t = _forward(batch, extractor, encoder, head, device, False)
                preds.append(p)
                targets.append(t)
            P, T_ = torch.cat(preds), torch.cat(targets)
            ccc_v = ccc(P[:, 0], T_[:, 0]).item()
            ccc_a = ccc(P[:, 1], T_[:, 1]).item()
            mae = (P - T_).abs().mean().item()
        mean_ccc = 0.5 * (ccc_v + ccc_a)
        flag = ""
        if mean_ccc > best:
            best = mean_ccc
            torch.save({
                "head": head.state_dict(), "enc_dim": enc_dim,
                "asr_checkpoint": str(args.checkpoint), "audio_cfg": audio_cfg,
                "val_ccc_v": ccc_v, "val_ccc_a": ccc_a, "val_mae": mae, "epoch": ep,
                "data_license": args.data_license,
            }, args.out)
            flag = "  <- best zapisany"
        print(f"[ep {ep:2d}] loss {np.mean(losses):.3f} | val CCC V {ccc_v:.3f} "
              f"A {ccc_a:.3f} | MAE {mae:.3f} | {time.time()-t0:.0f}s{flag}")
    print(f"[OK] best mean CCC {best:.3f} -> {args.out} ({args.data_license})")


if __name__ == "__main__":
    main()
