#!/usr/bin/env python3
"""
Pseudo-labeling (self-training): dekoduje audio checkpointem ASR i zapisuje
manifest JSONL {"audio_path", "duration", "text": hipoteza, "pl_score": ...}
przefiltrowany po pewnosci -- do dolozenia do miksu treningowego.

    python utils/pseudo_label.py --checkpoint checkpoints/conformer_big/<exp>/best.pt \
        --manifest data/train_manifest_yodas_extra.jsonl --out data/pseudo_yodas.jsonl \
        [--min-score -0.5] [--min-chars 3] [--max-tokens-per-sec 0]

Manifest wejsciowy potrzebuje tylko audio_path + duration (pole text ignorowane).
Score = sredni log-prob wyemitowanych tokenow z dekodowania greedy (proxy
pewnosci); prog --min-score np. -0.3 (ostry) ... -0.7 (luzny). Typowy przeplyw:
dociagnac wiecej shardow YODAS (hf2json.py yodas --shards N), zdekodowac je
najlepszym modelem, przefiltrowac i dokleic wynik do train_manifest_all.jsonl.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from model import ConformerTransducer  # noqa: E402
from preprocessing import (  # noqa: E402
    AudioConfig,
    CharTokenizer,
    LogMelExtractor,
    SentencePieceTokenizer,
    cmvn,
    subsampled_length,
)
from train import LengthBucketBatchSampler, greedy_decode_batch  # noqa: E402


class _AudioOnlyDS(Dataset):
    """Manifest bez transkrypcji -> cechy log-mel. Wadliwy plik -> features=None."""

    def __init__(self, entries: list[dict], audio_cfg: AudioConfig):
        self.entries = entries
        self.extractor = LogMelExtractor(audio_cfg)
        self.durations = [e.get("duration", 0.0) for e in entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        try:
            feats = cmvn(self.extractor(self.entries[i]["audio_path"]))
        except Exception as e:
            print(f"[WARN] pomijam {self.entries[i]['audio_path']}: {e}", flush=True)
            return {"idx": i, "features": None}
        return {"idx": i, "features": torch.from_numpy(feats)}


def _collate(batch: list[dict]) -> dict | None:
    batch = [b for b in batch if b["features"] is not None]
    if not batch:
        return None
    feats = [b["features"] for b in batch]
    lens = [f.shape[0] for f in feats]
    T_max = max(lens)
    padded = torch.zeros(len(batch), T_max, feats[0].shape[1], dtype=torch.float32)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    logit_lens = torch.tensor([subsampled_length(L) for L in lens], dtype=torch.int32)
    mask = torch.arange(int(logit_lens.max()))[None, :] >= logit_lens[:, None]
    return {
        "audio_features": padded,
        "logit_lengths": logit_lens,
        "audio_padding_mask": mask,
        "idx": [b["idx"] for b in batch],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Pseudo-labeling nieopisanego audio")
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--min-score",
        type=float,
        default=-0.5,
        help="Prog sredniego log-prob tokenow (wyzszy = ostrzejszy filtr)",
    )
    ap.add_argument("--min-chars", type=int, default=3)
    ap.add_argument(
        "--max-tokens-per-sec",
        type=float,
        default=0.0,
        help="Odetnij podejrzanie geste hipotezy (petle); 0 = wylaczone",
    )
    ap.add_argument("--min-duration", type=float, default=0.5)
    ap.add_argument("--max-duration", type=float, default=20.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-batch-duration", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    state = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    tok_meta = state["tokenizer"]
    if tok_meta["type"] == "bpe":
        tokenizer = SentencePieceTokenizer(tok_meta["bpe_model_path"])
    else:
        tokenizer = CharTokenizer(alphabet=tok_meta["alphabet"])
    audio_cfg = AudioConfig(**state["audio_cfg"])
    model = ConformerTransducer(
        vocab_size=tokenizer.vocab_size, n_mels=audio_cfg.n_mels, **state["model_cfg"]
    )
    model.load_state_dict(state["model"])
    model.eval().to(device)
    max_symbols = int(state.get("cfg", {}).get("max_symbols_per_step", 3))

    entries = []
    n_dur = 0
    with args.manifest.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            e = json.loads(line)
            d = e.get("duration", 0.0)
            if d < args.min_duration or d > args.max_duration:
                n_dur += 1
                continue
            entries.append(e)
    entries.sort(key=lambda e: e.get("duration", 0.0))
    if args.limit:
        entries = entries[: args.limit]
    print(f"[INFO] {len(entries)} sampli do dekodowania ({n_dur} odrzucone duracja)")

    ds = _AudioOnlyDS(entries, audio_cfg)
    sampler = LengthBucketBatchSampler(
        ds.durations,
        args.batch_size,
        shuffle=False,
        max_batch_duration=args.max_batch_duration,
    )
    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=_collate,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    use_amp = device.type == "cuda"
    kept = drop_score = drop_chars = drop_rate = 0
    t0 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as out_f, torch.no_grad():
        for bi, batch in enumerate(loader):
            if batch is None:
                continue
            audio = batch["audio_features"].to(device)
            logit_lengths = batch["logit_lengths"].to(device)
            mask = batch["audio_padding_mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                hyp_ids, scores = greedy_decode_batch(
                    model,
                    audio,
                    logit_lengths,
                    audio_padding_mask=mask,
                    max_symbols_per_step=max_symbols,
                    return_scores=True,
                )
            for ids, score, idx in zip(hyp_ids, scores, batch["idx"]):
                entry = entries[idx]
                text = tokenizer.decode(ids)
                dur = float(entry.get("duration", 0.0))
                if score < args.min_score:
                    drop_score += 1
                    continue
                if len(text) < args.min_chars:
                    drop_chars += 1
                    continue
                if (
                    args.max_tokens_per_sec > 0
                    and dur > 0
                    and len(ids) / dur > args.max_tokens_per_sec
                ):
                    drop_rate += 1
                    continue
                out_f.write(
                    json.dumps(
                        {
                            "audio_path": entry["audio_path"],
                            "duration": dur,
                            "text": text,
                            "pl_score": round(float(score), 4),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                kept += 1
            if bi % 20 == 0:
                done = kept + drop_score + drop_chars + drop_rate
                print(
                    f"  {done}/{len(entries)} | zostaje {kept} | "
                    f"{done / max(1e-9, time.time() - t0):.0f} sampli/s",
                    flush=True,
                )

    total = kept + drop_score + drop_chars + drop_rate
    print(f"[OK] {kept}/{total} pseudo-labeli -> {args.out}")
    print(
        f"     odrzucone: {drop_score} score < {args.min_score}, "
        f"{drop_chars} za krotkie, {drop_rate} tokens/s"
    )


if __name__ == "__main__":
    main()
