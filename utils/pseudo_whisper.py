#!/usr/bin/env python3
"""
FAZA 2 (destylacja): etykietowanie segmentow (z longform2json.py) Whisperem
large-v3 -> manifest treningowy {"audio_path", "duration", "text"}. Przejmujemy
wiedze wiekszego modelu (1550M) do naszego -- klasyczna destylacja przez pseudo-labels.

    python utils/pseudo_whisper.py --manifest data/pseudo_yt/manifest_nolabel.jsonl \
        --out data/pseudo_yt/manifest_whisper.jsonl [--batch-size 16]

Batchowane HF transformers (openai/whisper-large-v3, bf16) -- czysty torch, dziala
na Blackwell (sm_120) i H100; faster-whisper/ctranslate2 celowo NIE (prebudowane
kernele CUDA bez sm_120). Segmenty <= 18 s < okno 30 s Whispera -> bez chunkingu.

Filtry jakosci (raport na koncu): puste/za krotkie, tempo znakow poza [min,max]
(halucynacje/szum), powtorzenia n-gramow (klasyczna petla Whispera), watermarki
napisow YT (amara.org itp.), cyfry (nasz filtr treningowy i tak je tnie -- taniej
odrzucic tu; --keep-digits wylacza). Tekst zostaje z casingiem/interpunkcja (jak
CV/MLS; normalize_text robi swoje przy treningu).

Wznawialnosc: sample obecne w --out sa pomijane. NIE odpalac na podzie w trakcie
treningu. Zaleznosci: requirements-pseudo.txt (transformers).
"""
import argparse
import json
import re
import sys
from pathlib import Path

# watermarki spolecznosci napisow YT -- Whisper wypluwa je na muzyce/ciszy
_HALLUCINATION_RE = re.compile(
    r"amara\.org|napisy stworzone przez|napisy wykonane przez|zsub|subtitles by",
    re.IGNORECASE,
)


def quality_check(
    text: str,
    duration: float,
    min_cps: float = 2.0,
    max_cps: float = 30.0,
    keep_digits: bool = False,
) -> tuple[bool, str]:
    """(ok, powod_odrzucenia) dla pseudo-etykiety. Progi cps: zdrowa polszczyzna
    ~10-18 zn/s; ponizej 2 = halucynacja na ciszy, powyzej 30 = smieci/petla."""
    t = text.strip()
    if len(t) < 3:
        return False, "pusty"
    if _HALLUCINATION_RE.search(t):
        return False, "watermark"
    if not keep_digits and any(c.isdigit() for c in t):
        return False, "cyfry"
    cps = len(t) / max(duration, 0.1)
    if cps < min_cps:
        return False, "cps_niski"
    if cps > max_cps:
        return False, "cps_wysoki"
    words = t.lower().split()
    if len(words) >= 8:
        trigrams = [tuple(words[i: i + 3]) for i in range(len(words) - 2)]
        if len(set(trigrams)) / len(trigrams) < 0.5:  # petla dekodera
            return False, "powtorzenia"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Pseudo-etykiety Whisperem (destylacja)")
    ap.add_argument("--manifest", required=True, type=Path, help="manifest_nolabel.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--batch-size", type=int, default=16, help="H100: 32+; RTX 5080: 8-16")
    ap.add_argument("--device", default=None, help="cuda/cpu (auto)")
    ap.add_argument("--limit", type=int, default=None, help="max sampli (pilot)")
    ap.add_argument("--min-cps", type=float, default=2.0)
    ap.add_argument("--max-cps", type=float, default=30.0)
    ap.add_argument("--keep-digits", action="store_true")
    args = ap.parse_args()

    try:
        import soundfile as sf
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    except ImportError as e:
        sys.exit(f"brak zaleznosci ({e}) -- pip install -r requirements-pseudo.txt")
    from tqdm import tqdm

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    with open(args.manifest, encoding="utf-8") as f:
        entries = [json.loads(ln) for ln in f if ln.strip()]
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(ln)["audio_path"] for ln in f if ln.strip()}
        print(f"[resume] {len(done)} sampli juz zrobionych")
    entries = [e for e in entries if e["audio_path"] not in done]
    if args.limit:
        entries = entries[: args.limit]
    if not entries:
        sys.exit("nic do zrobienia")
    entries.sort(key=lambda e: e["duration"])  # podobne dlugosci w batchu -> mniej paddingu

    print(f"[whisper] laduje {args.model} na {device} ({dtype})...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device).eval()

    kept = 0
    kept_h = 0.0
    dropped: dict[str, int] = {}
    with open(args.out, "a", encoding="utf-8") as out_f, torch.no_grad():
        for i in tqdm(range(0, len(entries), args.batch_size), desc="batche",
                      dynamic_ncols=True, ascii=True):
            batch = entries[i: i + args.batch_size]
            audios = []
            for e in batch:
                audio, sr = sf.read(e["audio_path"], dtype="float32")
                assert sr == 16000, f"oczekiwane 16 kHz, jest {sr}: {e['audio_path']}"
                audios.append(audio)
            feats = processor(
                audios, sampling_rate=16000, return_tensors="pt",
            ).input_features.to(device, dtype)
            ids = model.generate(
                feats, language="pl", task="transcribe", max_new_tokens=180,
            )
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            for e, text in zip(batch, texts):
                ok, reason = quality_check(
                    text, e["duration"], args.min_cps, args.max_cps, args.keep_digits,
                )
                if not ok:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                out_f.write(json.dumps(
                    {"audio_path": e["audio_path"], "duration": e["duration"],
                     "text": text.strip()},
                    ensure_ascii=False,
                ) + "\n")
                kept += 1
                kept_h += e["duration"] / 3600
            out_f.flush()

    n_drop = sum(dropped.values())
    print(f"\n[OK] {kept} sampli / {kept_h:.1f} h -> {args.out}")
    print(f"odrzucone {n_drop} ({n_drop / max(1, kept + n_drop):.0%}): "
          + ", ".join(f"{k}={v}" for k, v in sorted(dropped.items())))
    if n_drop / max(1, kept + n_drop) > 0.3:
        print("[uwaga] >30% odrzutow -- obejrzyj probki zrodla, moze audio nie-polskie/muzyka")


if __name__ == "__main__":
    main()
