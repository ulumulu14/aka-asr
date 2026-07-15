#!/usr/bin/env python3
"""
Trenuje tokenizer BPE (sentencepiece) na transkrypcjach z manifestow JSONL.

Uzycie:
    python utils/train_bpe.py train_manifest.jsonl [inne_manifesty.jsonl ...] \
        --vocab-size 512 --model-prefix bpe_pl

Potem w configu:
    tokenizer:
      type: bpe
      bpe_model_path: bpe_pl.model
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

# repo root na sciezce, zeby zaimportowac preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing import normalize_text  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening tokenizera BPE (sentencepiece)")
    parser.add_argument("manifests", nargs="+", type=Path, help="Manifesty JSONL z polem 'text'")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--model-prefix", default="bpe_pl", help="Prefiks plikow .model/.vocab")
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ImportError:
        sys.exit("Brakuje sentencepiece: pip install sentencepiece")

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        n_lines = 0
        for manifest in args.manifests:
            with manifest.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    text = normalize_text(json.loads(line)["text"])
                    if text:
                        tmp.write(text + "\n")
                        n_lines += 1
        corpus_path = tmp.name

    print(f"[INFO] Korpus: {n_lines} linii")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=1.0,  # polskie diakrytyki musza byc pokryte
        bos_id=-1,
        eos_id=-1,
        unk_id=0,
    )
    Path(corpus_path).unlink(missing_ok=True)
    print(f"[OK] Zapisano {args.model_prefix}.model i {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()
