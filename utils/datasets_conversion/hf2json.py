#!/usr/bin/env python3
"""
hf2json.py — pobieranie polskich datasetow ASR z Hugging Face i konwersja do
manifestow JSONL w formacie ASRDataset ({"audio_path": <abs>, "duration": <s>,
"text": <str>} per linia).

Zrodla (subkomendy):
  bigos      amu-cai/pl-asr-bigos-v2 — Clarin studio/mobile, M-AILABS, FLEURS,
             minds14, korpusy PWR (bez fair-mls i common_voice: te mamy z oryginalow).
             Dataset gated (auto) -> wymaga HF_TOKEN i jednorazowej akceptacji
             warunkow na https://huggingface.co/datasets/amu-cai/pl-asr-bigos-v2
  voxpopuli  facebook/voxpopuli, config 'pl' (~100 h, Parlament Europejski);
             parquety z eksportu HF pobierane na dysk i czytane pyarrow batchami
             (streaming `datasets` buforowal caly plik w RAM), audio -> wav lokalnie
  yodas      espnet/yodas, subset pl000 (manualne napisy z YouTube);
             --shards ogranicza wolumen (1 shard ~ 0.9 GB audio)

Przyklady:
  python utils/datasets_conversion/hf2json.py bigos
  python utils/datasets_conversion/hf2json.py voxpopuli
  python utils/datasets_conversion/hf2json.py yodas --shards 10

Manifesty laduja w --manifest-dir (domyslnie data/):
  {train,val}_manifest_<zrodlo>.jsonl (+ test_manifest_bigos.jsonl gdy refy niepuste)
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

BIGOS_REPO = "amu-cai/pl-asr-bigos-v2"
YODAS_REPO = "espnet/yodas"

# fair-mls-20 i mozilla-common_voice_15-23 pominiete: MLS i CV mamy z oryginalow
BIGOS_SUBSETS = [
    "google-fleurs-22",
    "mailabs-corpus_librivox-19",
    "pjatk-clarin_mobile-15",
    "pjatk-clarin_studio-15",
    "polyai-minds14-21",
    "pwr-azon_read-20",
    "pwr-azon_spont-20",
    "pwr-maleset-unk",
    "pwr-shortwords-unk",
    "pwr-viu-unk",
]

BIGOS_TEXT_COLUMNS = ("ref_orig", "ref", "sentence", "text", "transcription")
# audiopath_bigos ma pelna nazwe pliku (.wav); audioname jest bez rozszerzenia
BIGOS_AUDIO_COLUMNS = ("audiopath_bigos", "audioname", "path", "file", "audio", "wav")


def token() -> str | None:
    return os.environ.get("HF_TOKEN")


def safe_extract_tar_gz(tar_path: Path, output_dir: Path) -> None:
    """
    Rozpakowanie z ochrona przed path traversal (por. cv2json.py). Strumieniowo,
    jeden przebieg: walidacja + ekstrakcja per member (getmembers()+extractall()
    dekompresowaly gzip 2x i szly po cichu -- na network volume wygladalo jak zwis).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tqdm(tar, desc="ekstrakcja", unit="plik"):
            target = (output_dir / member.name).resolve()
            if os.path.commonpath([output_root, target]) != str(output_root):
                raise RuntimeError(f"Niebezpieczna sciezka w archiwum: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archiwum zawiera link: {member.name}")
            tar.extract(member, output_dir)


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    hours = sum(e["duration"] for e in entries) / 3600
    print(f"[OK] {path}: {len(entries)} sampli, {hours:.1f} h")


def probe_durations(paths: list[Path], workers: int = 8) -> list[float | None]:
    def one(p: Path) -> float | None:
        try:
            return round(sf.info(str(p)).duration, 3)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(tqdm(pool.map(one, paths), total=len(paths), desc="duration", leave=False))


# ---------------------------------------------------------------- BIGOS


def convert_bigos(args: argparse.Namespace) -> None:
    import csv

    out_root = args.out_root / "bigos"
    per_split: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for subset in args.subsets:
        for split in ("train", "validation", "test"):
            print(f"[bigos] {subset}/{split}")
            try:
                tsv = hf_hub_download(
                    BIGOS_REPO,
                    f"data/{subset}/{split}.tsv",
                    repo_type="dataset",
                    token=token(),
                )
            except Exception as e:
                print(f"[WARN] brak {subset}/{split}.tsv ({e}) -- pomijam")
                continue

            audio_dir = out_root / subset / split
            if not audio_dir.exists() or not any(audio_dir.rglob("*")):
                tar = hf_hub_download(
                    BIGOS_REPO,
                    f"data/{subset}/{split}.tar.gz",
                    repo_type="dataset",
                    token=token(),
                    local_dir=args.out_root / "_hf_tmp",
                )
                safe_extract_tar_gz(Path(tar), audio_dir)
                Path(tar).unlink(missing_ok=True)  # tar niepotrzebny po ekstrakcji
            else:
                print(f"       {audio_dir} juz istnieje -- pomijam pobieranie")

            by_name = {p.name: p for p in audio_dir.rglob("*") if p.is_file()}

            with open(tsv, encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                cols = reader.fieldnames or []
                text_col = next((c for c in BIGOS_TEXT_COLUMNS if c in cols), None)
                audio_col = next((c for c in BIGOS_AUDIO_COLUMNS if c in cols), None)
                if text_col is None or audio_col is None:
                    raise RuntimeError(f"Nieznane kolumny TSV {subset}/{split}: {cols}")
                rows = [
                    (r[audio_col].strip(), r[text_col].strip())
                    for r in reader
                    if r.get(audio_col) and (r.get(text_col) or "").strip()
                ]

            paths, texts = [], []
            n_missing = 0
            for name, text in rows:
                base = Path(name).name
                p = by_name.get(base) or by_name.get(base + ".wav")
                if p is None:
                    n_missing += 1
                    continue
                paths.append(p)
                texts.append(text)
            if n_missing:
                print(f"[WARN] {subset}/{split}: {n_missing} plikow z TSV nie ma w tar")

            durations = probe_durations(paths)
            for p, text, dur in zip(paths, texts, durations):
                if dur is None:
                    continue
                per_split[split].append(
                    {"audio_path": str(p.resolve()), "duration": dur, "text": text}
                )

    write_manifest(per_split["train"], args.manifest_dir / "train_manifest_bigos.jsonl")
    write_manifest(per_split["validation"], args.manifest_dir / "val_manifest_bigos.jsonl")
    if per_split["test"]:
        write_manifest(per_split["test"], args.manifest_dir / "test_manifest_bigos.jsonl")
    else:
        print("[INFO] test: refy puste/ukryte (benchmark BIGOS) -- brak manifestu test")


# ---------------------------------------------------------------- VoxPopuli

VOXPOPULI_REPO = "facebook/voxpopuli"
# PRZYPIETY commit brancha refs/convert/parquet (stan 2026-07-11). Sam branch jest
# MUTOWALNY -- regeneracja po 2026-07-11 zepsula normalized_text dla pl (diakrytyki
# zamienione na "_"), co wykrylismy dopiero w targetach treningu. Nie wracac do
# "refs/convert/parquet" bez sprawdzenia tekstu!
VOXPOPULI_PARQUET_REV = "930c9a00c3aeff80a6b9fbb1175bcd07bf751cf8"


def _voxpopuli_parquet_paths(split: str) -> list[str]:
    from huggingface_hub import HfApi

    tree = HfApi().list_repo_tree(
        VOXPOPULI_REPO,
        repo_type="dataset",
        revision=VOXPOPULI_PARQUET_REV,
        path_in_repo=f"pl/{split}",
        recursive=True,
    )
    return sorted(f.path for f in tree if f.path.endswith(".parquet"))


def convert_voxpopuli(args: argparse.Namespace) -> None:
    # Streaming przez `datasets` buforuje pobierane parquety w RAM bez limitu
    # (OOM na 16 GB), wiec: parquet na dysk -> pyarrow batchami -> kasujemy.
    # Fragment manifestu per parquet = wznawialnosc bez ponownego pobierania.
    import pyarrow.parquet as pq

    for split, manifest_name in (
        ("train", "train_manifest_voxpopuli.jsonl"),
        ("validation", "val_manifest_voxpopuli.jsonl"),
    ):
        out_dir = args.out_root / "voxpopuli_pl" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        fragments: list[Path] = []

        for rpath in _voxpopuli_parquet_paths(split):
            frag = out_dir / f".manifest_{Path(rpath).stem}.jsonl"
            fragments.append(frag)
            if frag.exists():
                print(f"[voxpopuli] {rpath}: fragment istnieje -- pomijam")
                continue

            local = Path(
                hf_hub_download(
                    VOXPOPULI_REPO,
                    rpath,
                    repo_type="dataset",
                    revision=VOXPOPULI_PARQUET_REV,
                    local_dir=args.out_root / "_hf_tmp",
                )
            )
            entries: list[dict] = []
            pf = pq.ParquetFile(local)
            columns = [
                c
                for c in ("audio", "normalized_text", "raw_text", "audio_id")
                if c in pf.schema_arrow.names
            ]
            for batch in pf.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    text = (row.get("normalized_text") or "").strip()
                    # zepsuta normalizacja pl podmienia diakrytyki na "_" -> raw_text
                    # (normalize_text w treningu i tak zrobi lowercase)
                    if "_" in text:
                        text = (row.get("raw_text") or "").strip() or text
                    if not text:
                        text = (row.get("raw_text") or "").strip()
                    if not text:
                        continue
                    audio = row["audio"]
                    raw: bytes = audio["bytes"]
                    name = Path(audio.get("path") or f"{row['audio_id']}.wav").name
                    out_path = out_dir / name
                    if not out_path.exists():
                        out_path.write_bytes(raw)
                    try:
                        dur = round(sf.info(io.BytesIO(raw)).duration, 3)
                    except Exception:
                        continue
                    entries.append(
                        {"audio_path": str(out_path.resolve()), "duration": dur, "text": text}
                    )
            pf.close()
            write_manifest(entries, frag)
            local.unlink(missing_ok=True)

        manifest_path = args.manifest_dir / manifest_name
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        n, hours = 0, 0.0
        with manifest_path.open("w", encoding="utf-8") as out:
            for frag in fragments:
                with frag.open(encoding="utf-8") as f:
                    for line in f:
                        out.write(line)
                        n += 1
                        hours += json.loads(line)["duration"] / 3600
        print(f"[OK] {manifest_path}: {n} sampli, {hours:.1f} h")


# ---------------------------------------------------------------- YODAS


def _read_kv(path: Path) -> dict[str, str]:
    """Pliki yodas text/duration: '<utt_id> <wartosc do konca linii>'."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def convert_yodas(args: argparse.Namespace) -> None:
    out_root = args.out_root / "yodas_pl"
    entries: list[dict] = []

    for i in range(args.shards):
        shard = f"{i:08d}"
        print(f"[yodas] shard {shard} ({i + 1}/{args.shards})")
        texts = _read_kv(
            Path(hf_hub_download(YODAS_REPO, f"data/pl000/text/{shard}.txt", repo_type="dataset"))
        )
        durs = _read_kv(
            Path(
                hf_hub_download(YODAS_REPO, f"data/pl000/duration/{shard}.txt", repo_type="dataset")
            )
        )

        audio_dir = out_root / shard
        if not audio_dir.exists() or not any(audio_dir.rglob("*")):
            tar = hf_hub_download(
                YODAS_REPO,
                f"data/pl000/audio/{shard}.tar.gz",
                repo_type="dataset",
                local_dir=args.out_root / "_hf_tmp",
            )
            safe_extract_tar_gz(Path(tar), audio_dir)
            Path(tar).unlink(missing_ok=True)
        else:
            print(f"       {audio_dir} juz istnieje -- pomijam pobieranie")

        n_before = len(entries)
        for p in audio_dir.rglob("*"):
            if not p.is_file():
                continue
            utt = p.stem
            text = texts.get(utt, "").strip()
            if not text:
                continue
            dur_s = durs.get(utt)
            try:
                dur = round(float(dur_s), 3) if dur_s else round(sf.info(str(p)).duration, 3)
            except Exception:
                continue
            entries.append({"audio_path": str(p.resolve()), "duration": dur, "text": text})
        print(f"       +{len(entries) - n_before} sampli")

    write_manifest(entries, args.manifest_dir / "train_manifest_yodas.jsonl")


# ---------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--out-root", type=Path, default=Path("data"), help="Katalog na audio (domyslnie data/)"
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("data"),
        help="Katalog na manifesty JSONL (domyslnie data/)",
    )
    sub = parser.add_subparsers(dest="source", required=True)

    p_bigos = sub.add_parser("bigos", help="amu-cai/pl-asr-bigos-v2")
    p_bigos.add_argument("--subsets", nargs="+", default=BIGOS_SUBSETS)
    p_bigos.set_defaults(fn=convert_bigos)

    p_vox = sub.add_parser("voxpopuli", help="facebook/voxpopuli 'pl'")
    p_vox.set_defaults(fn=convert_voxpopuli)

    p_yodas = sub.add_parser("yodas", help="espnet/yodas pl000 (manualne napisy)")
    p_yodas.add_argument(
        "--shards", type=int, default=10, help="Ile shardow pobrac (1 shard ~ 0.9 GB; calosc = 78)"
    )
    p_yodas.set_defaults(fn=convert_yodas)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
