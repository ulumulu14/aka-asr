#!/usr/bin/env python3
"""
FAZA 2 (destylacja): dlugie nieopisane audio (podcasty, Sejm, YT z yt-dlp) ->
segmenty mowy 3-18 s (silero VAD) -> wav 16 kHz mono PCM16 + manifest JSONL
BEZ tekstu ({"audio_path", "duration", "text": ""}). Etykiety dorabia potem
utils/pseudo_whisper.py (Whisper large-v3) albo utils/pseudo_label.py (nasz model).

    python utils/datasets_conversion/longform2json.py IN_DIR \
        --out-root data/pseudo_yt [--min-seg 3] [--max-seg 18]

Wejscie: dowolne formaty audio/wideo (mp3/m4a/opus/webm/mp4/wav/...) -- dekoduje
ffmpeg (wymagany w PATH). Wznawialnosc: ukonczone zrodla w <out>/done_sources.txt,
ponowny run pomija. Segmenty per zrodlo w podkatalogu (network volume nie lubi
tysiecy plikow w jednym katalogu).

Zaleznosci: requirements-pseudo.txt (silero-vad). NIE odpalac na podzie w trakcie
treningu (I/O + CPU walcza z dataloaderem).
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

AUDIO_EXTS = {".mp3", ".m4a", ".opus", ".webm", ".mp4", ".mkv", ".wav", ".flac", ".ogg", ".aac"}
SR = 16000


def pack_spans(
    spans: list[tuple[float, float]],
    min_seg: float = 3.0,
    max_seg: float = 18.0,
    merge_gap: float = 0.5,
) -> list[tuple[float, float]]:
    """Spany mowy z VAD -> segmenty treningowe [min_seg, max_seg].

    1) scala sasiednie spany gdy przerwa <= merge_gap i laczna dlugosc <= max_seg
       (krotkie frazy w naturalne zdania),
    2) tnie dluzsze niz max_seg na ROWNE kawalki (zaden nie wypada mikro-ogonem),
    3) wyrzuca finalnie krotsze niz min_seg (izolowane wtracenia/szum).
    """
    if not spans:
        return []
    merged: list[list[float]] = [list(spans[0])]
    for s, e in spans[1:]:
        last = merged[-1]
        if s - last[1] <= merge_gap and e - last[0] <= max_seg:
            last[1] = e
        else:
            merged.append([s, e])

    out: list[tuple[float, float]] = []
    for s, e in merged:
        dur = e - s
        if dur <= max_seg:
            pieces = 1
        else:
            pieces = int(dur // max_seg) + (1 if dur % max_seg else 0)
        step = dur / pieces
        for i in range(pieces):
            a, b = s + i * step, s + (i + 1) * step
            if b - a >= min_seg:
                out.append((a, b))
    return out


def _safe_stem(path: Path) -> str:
    stem = re.sub(r"[^0-9A-Za-z_-]+", "_", path.stem).strip("_")
    return stem[:80] or "audio"


def _decode_to_wav(src: Path, tmp_dir: Path) -> Path:
    """Dowolny format -> tymczasowy wav 16 kHz mono (ffmpeg)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir, delete=False)
    tmp.close()
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR), str(tmp.name)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg: {res.stderr.strip()[:200]}")
    return Path(tmp.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dlugie audio -> segmenty VAD + manifest bez tekstu")
    ap.add_argument("in_dir", type=Path, help="katalog ze zrodlowym audio (rekurencyjnie)")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--min-seg", type=float, default=3.0)
    ap.add_argument("--max-seg", type=float, default=18.0)
    ap.add_argument("--merge-gap", type=float, default=0.5)
    ap.add_argument("--limit-sources", type=int, default=None, help="max plikow zrodlowych (pilot)")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg nie znaleziony w PATH (apt install ffmpeg)")
    try:
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad
    except ImportError as e:
        sys.exit(f"brak zaleznosci ({e}) -- pip install -r requirements-pseudo.txt")
    from tqdm import tqdm

    out_root = args.out_root
    (out_root / "wav").mkdir(parents=True, exist_ok=True)
    tmp_dir = out_root / ".tmp"
    tmp_dir.mkdir(exist_ok=True)
    done_path = out_root / "done_sources.txt"
    done = set(done_path.read_text().splitlines()) if done_path.exists() else set()
    manifest_path = out_root / "manifest_nolabel.jsonl"

    sources = sorted(p for p in args.in_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS)
    sources = [p for p in sources if str(p.resolve()) not in done]
    if args.limit_sources:
        sources = sources[: args.limit_sources]
    if not sources:
        sys.exit("brak nowych zrodel (wszystko w done_sources.txt?)")

    vad = load_silero_vad()
    n_seg = 0
    total_h = 0.0
    with open(manifest_path, "a", encoding="utf-8") as mf:
        for src in tqdm(sources, desc="zrodla", dynamic_ncols=True, ascii=True):
            try:
                wav_path = _decode_to_wav(src, tmp_dir)
            except RuntimeError as e:
                tqdm.write(f"[skip] {src.name}: {e}")
                continue
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
                assert sr == SR
                spans_raw = get_speech_timestamps(
                    torch.from_numpy(audio), vad, sampling_rate=SR, return_seconds=True,
                )
                spans = pack_spans(
                    [(t["start"], t["end"]) for t in spans_raw],
                    args.min_seg, args.max_seg, args.merge_gap,
                )
                stem = _safe_stem(src)
                seg_dir = out_root / "wav" / stem
                seg_dir.mkdir(exist_ok=True)
                for i, (a, b) in enumerate(spans):
                    seg = audio[int(a * SR): int(b * SR)]
                    seg_path = seg_dir / f"{stem}_{i:04d}.wav"
                    sf.write(seg_path, seg, SR, subtype="PCM_16")
                    entry = {"audio_path": str(seg_path.resolve()),
                             "duration": round(len(seg) / SR, 3), "text": ""}
                    mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    n_seg += 1
                    total_h += len(seg) / SR / 3600
                mf.flush()
                # zrodlo domkniete w calosci -> odporne na przerwanie miedzy plikami
                with open(done_path, "a", encoding="utf-8") as df:
                    df.write(str(src.resolve()) + "\n")
            finally:
                wav_path.unlink(missing_ok=True)

    print(f"[OK] {n_seg} segmentow / {total_h:.1f} h -> {manifest_path}")
    print("nastepny krok: python utils/pseudo_whisper.py --manifest "
          f"{manifest_path} --out {out_root}/manifest_whisper.jsonl")


if __name__ == "__main__":
    main()
