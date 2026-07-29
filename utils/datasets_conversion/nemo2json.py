#!/usr/bin/env python3
"""
nEMO (amu-cai/nEMO) -> wav 16k PCM16 + manifest affect:
    {"audio_path", "duration", "emotion", "valence", "arousal", "speaker_id", "text"}

** LICENCJA CC-BY-NC-SA: dane TYLKO do prototypu/ewaluacji, NIE do wag komercyjnych. **

    python utils/datasets_conversion/nemo2json.py --out-root data/nemo \
        --manifest data/nemo_affect.jsonl

V/A z mapowania kategorii (affect.head.EMOTION_VA). Wznawialne (pomija istniejace wav).
"""
import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

NEMO_REPO = "amu-cai/nEMO"
NEMO_SHARDS = ("data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet")
SR = 16000


def main() -> None:
    ap = argparse.ArgumentParser(description="nEMO -> manifest affect (V/A) [NC: prototyp]")
    ap.add_argument("--out-root", type=Path, default=Path("data/nemo"))
    ap.add_argument("--manifest", type=Path, default=Path("data/nemo_affect.jsonl"))
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from huggingface_hub import hf_hub_download

    from affect.head import EMOTION_VA

    print("** nEMO = CC-BY-NC-SA: wagi z tego treningu sa NIEkomercyjne (prototyp) **")
    wav_dir = args.out_root / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    hours = 0.0
    unknown: set[str] = set()
    with open(args.manifest, "w", encoding="utf-8") as mf:
        for shard in NEMO_SHARDS:
            p = hf_hub_download(NEMO_REPO, shard, repo_type="dataset")
            for batch in pq.ParquetFile(p).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    emo = row["emotion"]
                    if emo not in EMOTION_VA:
                        unknown.add(emo)
                        continue
                    out = wav_dir / f"{row['file_id']}.wav"
                    if not out.exists():
                        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                        if wav.ndim == 2:
                            wav = wav.mean(axis=1)
                        if sr != SR:
                            wav = AF.resample(
                                torch.from_numpy(wav).unsqueeze(0), sr, SR
                            ).squeeze(0).numpy()
                        sf.write(out, wav, SR, subtype="PCM_16")
                    dur = round(sf.info(str(out)).duration, 3)
                    v, a = EMOTION_VA[emo]
                    mf.write(json.dumps({
                        "audio_path": str(out.resolve()), "duration": dur,
                        "emotion": emo, "valence": v, "arousal": a,
                        "speaker_id": row["speaker_id"],
                        "text": (row.get("raw_text") or "").strip(),
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    hours += dur / 3600
    if unknown:
        print(f"[WARN] nieznane emocje pominiete: {sorted(unknown)} -- uzupelnij EMOTION_VA")
    print(f"[OK] {n} sampli / {hours:.2f} h -> {args.manifest}")


if __name__ == "__main__":
    main()
