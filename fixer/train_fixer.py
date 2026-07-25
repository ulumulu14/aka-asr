#!/usr/bin/env python3
"""
Fine-tune plT5 jako fixer ASR na parach {hyp -> ref} (z fixer/harvest.py).

Wejscie modelu = prefiks + hipoteza ASR, cel = poprawna transkrypcja. Zapisuje
model (save_pretrained) + fixer_meta.json (prefix, max_len) -> laduje go AutoFixer
w tescie i w inferencji (ta sama korekta w obu sciezkach).

    python -m fixer.train_fixer \
        --pairs fixer/pairs_train.jsonl [fixer/pairs_train_aug.jsonl ...] \
        --val-pairs fixer/pairs_val.jsonl \
        --base allegro/plt5-base --out checkpoints/fixer/plt5-base_v1 \
        --epochs 3 --batch-size 16

Zaleznosci: pip install -r requirements-fixer.txt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from fixer.fixer import DEFAULT_PREFIX  # noqa: E402


def read_pairs(paths, identity_keep: float = 1.0, seed: int = 42):
    rng = random.Random(seed)
    pairs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                hyp, ref = o.get("hyp", ""), o.get("ref", "")
                if not ref:
                    continue
                # opcjonalnie przerzedz pary hyp==ref (za duzo 'nic nie zmieniaj' rozleniwia)
                if hyp == ref and identity_keep < 1.0 and rng.random() > identity_keep:
                    continue
                pairs.append((hyp, ref))
    return pairs


class PairDS(Dataset):
    def __init__(self, pairs, prefix):
        self.pairs = pairs
        self.prefix = prefix

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        hyp, ref = self.pairs[i]
        return self.prefix + hyp, ref


def main() -> None:
    ap = argparse.ArgumentParser(description="Trening fixera (plT5 fine-tune)")
    ap.add_argument("--pairs", nargs="+", required=True, type=Path)
    ap.add_argument("--val-pairs", nargs="*", type=Path, default=[])
    ap.add_argument("--base", default="allegro/plt5-base")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument(
        "--identity-keep",
        type=float,
        default=1.0,
        help="Frakcja par hyp==ref do zachowania (1.0=wszystkie)",
    )
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from transformers import (  # noqa: E402
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    device = torch.device(args.device)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base).to(device)

    def collate(batch):
        src = [b[0] for b in batch]
        tgt = [b[1] for b in batch]
        enc = tok(src, max_length=args.max_len, truncation=True, padding=True, return_tensors="pt")
        lab = tok(
            text_target=tgt,
            max_length=args.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        labels = lab.input_ids
        labels[labels == tok.pad_token_id] = -100  # ignoruj padding w stracie
        enc["labels"] = labels
        return enc

    train_pairs = read_pairs(args.pairs, identity_keep=args.identity_keep)
    n_err = sum(h != r for h, r in train_pairs)
    print(
        f"[INFO] par treningowych: {len(train_pairs)} ({n_err} z bledem, "
        f"{len(train_pairs) - n_err} identycznych)"
    )
    train_loader = DataLoader(
        PairDS(train_pairs, args.prefix),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = None
    if args.val_pairs:
        val_loader = DataLoader(
            PairDS(read_pairs(args.val_pairs), args.prefix),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=args.num_workers,
        )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, args.warmup, total_steps)
    use_amp = device.type == "cuda"

    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            n += 1
            step += 1
            if step % 50 == 0:
                print(
                    f"  epoka {epoch} krok {step} loss={running / n:.4f} "
                    f"lr={sched.get_last_lr()[0]:.2e} {n / (time.time() - t0):.1f} it/s",
                    flush=True,
                )

        if val_loader:
            model.eval()
            vl, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
                    ):
                        vl += model(**batch).loss.item()
                    vn += 1
            print(f"[epoka {epoch}] val_loss={vl / max(1, vn):.4f}")
        _demo(model, tok, args, device)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "fixer_meta.json").write_text(
        json.dumps(
            {"base": args.base, "prefix": args.prefix, "max_len": args.max_len},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] zapisano fixer -> {args.out}")


def _demo(model, tok, args, device):
    """Kilka przykladowych korekt na koniec epoki (sanity)."""
    samples = ["ktoz by to zblicze", "dzien dobri panstwu", "sztuczna inteligencja rozpoznaje mowe"]
    model.eval()
    enc = tok([args.prefix + s for s in samples], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_length=args.max_len, num_beams=1)
    for s, o in zip(samples, tok.batch_decode(out, skip_special_tokens=True)):
        print(f"    {s!r} -> {o!r}")


if __name__ == "__main__":
    main()
