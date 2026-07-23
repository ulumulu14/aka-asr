#!/usr/bin/env python3
"""
Buduje manifest JSONL z Multilingual LibriSpeech (MLS).

Oczekiwana struktura (po rozpakowaniu mls_polish.tar.gz):
    mls_polish/
        train/
            transcripts.txt   # utt_id \t transkrypcja
            segments.txt      # utt_id \t url \t start \t end (opcjonalny)
            audio/{speaker}/{book}/{utt_id}.flac
        dev/ ...
        test/ ...

Konwersja do wav nie jest potrzebna -- torchaudio czyta flac bezposrednio,
a MLS jest juz w 16 kHz mono.

Uzycie:
    python librispeech2json.py /sciezka/do/mls_polish --split train --out train_manifest.jsonl
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def load_durations(split_dir: Path) -> dict[str, float]:
    """Dlugosci naglan z segments.txt (end - start); pusty dict gdy brak pliku."""
    segments_path = split_dir / "segments.txt"
    durations: dict[str, float] = {}
    if not segments_path.exists():
        return durations
    with segments_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            utt_id, _url, start, end = parts
            try:
                durations[utt_id] = round(float(end) - float(start), 3)
            except ValueError:
                continue
    return durations


def probe_duration(audio_path: Path) -> float:
    """Fallback: dlugosc z naglowka flac (bez dekodowania calego pliku)."""
    import soundfile as sf

    info = sf.info(str(audio_path))
    return round(info.frames / info.samplerate, 3)


def build_manifest(split_dir: Path, out_path: Path) -> None:
    transcripts_path = split_dir / "transcripts.txt"
    if not transcripts_path.exists():
        raise FileNotFoundError(f"Brak {transcripts_path} -- to na pewno katalog splitu MLS?")

    durations = load_durations(split_dir)
    if not durations:
        print("[INFO] Brak segments.txt -- dlugosci beda czytane z naglowkow flac (wolniejsze)")

    n_written = n_missing = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with transcripts_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in tqdm(fin, desc="budowanie manifestu"):
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            utt_id, text = line.split("\t", 1)
            text = text.strip()
            try:
                speaker, book, _ = utt_id.split("_")
            except ValueError:
                print(f"[WARN] Nieoczekiwany utt_id: {utt_id}")
                continue

            audio_path = split_dir / "audio" / speaker / book / f"{utt_id}.flac"
            if not audio_path.exists():
                n_missing += 1
                continue

            duration = durations.get(utt_id)
            if duration is None:
                duration = probe_duration(audio_path)

            fout.write(
                json.dumps(
                    {
                        "audio_path": str(audio_path.resolve()),
                        "duration": duration,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_written += 1

    print(f"[OK] Zapisano {n_written} rekordow do {out_path}")
    if n_missing:
        print(f"[WARN] Pominieto {n_missing} rekordow bez pliku audio")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zbuduj manifest JSONL z Multilingual LibriSpeech (MLS)."
    )
    parser.add_argument("mls_root", type=Path, help="Katalog MLS (zawiera train/dev/test)")
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Sciezka wyjsciowa (domyslnie {mls_root}/{split}_manifest.jsonl)",
    )
    args = parser.parse_args()

    split_dir = args.mls_root / args.split
    out_path = args.out or (args.mls_root / f"{args.split}_manifest.jsonl")
    build_manifest(split_dir, out_path)


if __name__ == "__main__":
    main()
