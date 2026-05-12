#!/usr/bin/env python3

import argparse
import csv
from itertools import islice
import json
import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from tqdm import tqdm

DATA_LIMIT = 25000


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            f"Brakuje programu '{command}'. Zainstaluj ffmpeg, np.:\n"
            f"  Ubuntu/Debian: sudo apt install ffmpeg\n"
            f"  macOS: brew install ffmpeg\n"
            f"  Windows: winget install Gyan.FFmpeg"
        )


def safe_extract_tar_gz(tar_path: Path, output_dir: Path) -> None:
    """
    Bezpieczne rozpakowanie tar.gz z ochroną przed path traversal,
    np. plikami typu ../../evil.sh.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_resolved = output_dir.resolve()

    print("Starting archive extraction...")

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar.getmembers()):
            target_path = (output_dir / member.name).resolve()

            if os.path.commonpath([output_dir_resolved, target_path]) != str(output_dir_resolved):
                raise RuntimeError(f"Niebezpieczna ścieżka w archiwum: {member.name}")

            if member.issym() or member.islnk():
                raise RuntimeError(f"Archiwum zawiera link symboliczny/twardy: {member.name}")

        tar.extractall(output_dir)
    
    print("Extraction finished.")


def find_tsv(extracted_dir: Path, split: str) -> Path:
    matches = sorted(extracted_dir.rglob(f"{split}.tsv"))

    if not matches:
        raise FileNotFoundError(
            f"Nie znaleziono pliku {split}.tsv w katalogu {extracted_dir}"
        )

    return matches[0]


def resolve_audio_path(tsv_path: Path, relative_audio_path: str) -> Path:
    """
    Common Voice zwykle ma strukturę:
      validated.tsv
      clips/common_voice_xxx.mp3

    W TSV kolumna 'path' zwykle zawiera samą nazwę pliku mp3.
    """
    parent = tsv_path.parent
    clips_dir = parent / "clips"

    candidates = [
        clips_dir / relative_audio_path,
        parent / relative_audio_path,
        parent / "clips" / Path(relative_audio_path).name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Nie znaleziono audio dla: {relative_audio_path}")


def convert_to_wav(
    input_audio: Path,
    output_wav: Path,
    sample_rate: int,
    mono: bool = True,
    overwrite: bool = False,
) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    if output_wav.exists() and not overwrite:
        return

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_audio),
    ]

    if mono:
        command += ["-ac", "1"]

    command += [
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(output_wav),
    ]

    subprocess.run(command, check=True)


def get_duration_seconds(audio_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    return round(float(result.stdout.strip()), 3)


def build_manifest(
    tsv_path: Path,
    wav_dir: Path,
    sample_rate: int,
    overwrite: bool,
) -> list[dict]:
    manifest = []
    
    with tsv_path.open("r", encoding="utf-8") as f:
        total = sum(1 for _ in f) - 1  # -1 na nagłówek

    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        required_columns = {"path", "sentence"}
        missing = required_columns - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(
                f"Brakuje kolumn w TSV: {missing}. "
                f"Dostępne kolumny: {reader.fieldnames}"
            )


        for i, row in enumerate(tqdm(reader, total=total), start=1):
            relative_audio_path = row["path"].strip()
            text = row["sentence"].strip()

            if not relative_audio_path or not text:
                continue
            if Path(relative_audio_path).is_dir():
                continue

            try:
                source_audio = resolve_audio_path(tsv_path, relative_audio_path)

                wav_filename = Path(relative_audio_path).with_suffix(".wav").name
                output_wav = wav_dir / wav_filename

                convert_to_wav(
                    input_audio=source_audio,
                    output_wav=output_wav,
                    sample_rate=sample_rate,
                    mono=True,
                    overwrite=overwrite,
                )

                duration = get_duration_seconds(output_wav)

                manifest.append(
                    {
                        "audio_path": str(output_wav.resolve()),
                        "duration": duration,
                        "text": text,
                    }
                )

            except Exception as e:
                print(f"[WARN] Pomijam rekord {i}: {e}")

    return manifest


def save_json(manifest: list[dict], output_path: Path, jsonl: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if jsonl:
        with output_path.open("w", encoding="utf-8") as f:
            for item in tqdm(manifest, desc='Saving JSON file'):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rozpakuj Mozilla Common Voice tar.gz i zbuduj manifest JSON."
    )

    parser.add_argument(
        "archive",
        type=Path,
        help="Ścieżka do pliku Common Voice .tar.gz",
    )

    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=Path("common_voice_extracted"),
        help="Katalog, do którego rozpakować dataset",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validated",
        choices=["validated", "train", "dev", "test", "other", "invalid"],
        help="Który plik TSV przetworzyć",
    )

    parser.add_argument(
        "--wav-dir",
        type=Path,
        default=Path("common_voice_wav"),
        help="Katalog wyjściowy na pliki WAV",
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("manifest.json"),
        help="Ścieżka wyjściowa manifestu JSON",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Docelowy sample rate WAV, np. 16000 albo 48000",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Nadpisuj istniejące pliki WAV",
    )

    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Zapisz jako JSON Lines zamiast jednej tablicy JSON",
    )

    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Nie rozpakowuj archiwum ponownie, użyj istniejącego --extract-dir",
    )

    args = parser.parse_args()

    require_command("ffmpeg")
    require_command("ffprobe")

    if not args.skip_extract:
        print(f"[INFO] Extractug: {args.archive}")
        safe_extract_tar_gz(args.archive, args.extract_dir)
    else:
        print(f"[INFO] Skipping data extraction, using: {args.extract_dir}")

    tsv_path = find_tsv(args.extract_dir, args.split)
    print(f"[INFO] Używam TSV: {tsv_path}")

    manifest = build_manifest(
        tsv_path=tsv_path,
        wav_dir=args.wav_dir,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
    )

    save_json(manifest, args.out, jsonl=args.jsonl)

    print(f"[OK] Zapisano {len(manifest)} rekordów do: {args.out}")
    print(f"[OK] Pliki WAV są w: {args.wav_dir}")


if __name__ == "__main__":
    main()