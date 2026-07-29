"""
transcriber.py - preprocessing audio + inferencja ConformerTransducer (RNN-T).

Wywoływane z Rusta przez PyO3:
    transcribe(samples: np.ndarray[frames, channels], sample_rate: int, ckpt_path: str) -> str

Checkpoint z train.py niesie własne metadane (model_cfg, tokenizer, audio_cfg),
więc model i front-end odtwarzane są bez configu YAML. Pipeline cech i dekodowanie
są identyczne z walidacją treningową:
    waveform -> mono -> resample 16k -> preemfaza -> log-mel -> CMVN
    -> encoder -> greedy transducer decode -> tekst
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Fallback dla uruchomień poza transcribe.rs (tam sys.path ustawia Rust):
# transcriber.py lezy w aka_asr_demo_app/, repo z modelem to katalog wyzej
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torchaudio.transforms as AT

from model import ConformerTransducer
from preprocessing import (
    AudioConfig,
    CharTokenizer,
    LogMelExtractor,
    SentencePieceTokenizer,
    cmvn,
    subsampled_length,
)
from train import greedy_decode_batch

# cpu domyslnie — demo nie wchodzi w droge treningowi na GPU; ASR_DEMO_DEVICE=cuda wymusza
DEVICE = os.environ.get("ASR_DEMO_DEVICE", "cpu")

_cache: dict[str, tuple] = {}


def _load_bundle(checkpoint_path: str) -> tuple:
    """(model, tokenizer, extractor, max_symbols_per_step) z metadanych checkpointa."""
    bundle = _cache.get(checkpoint_path)
    if bundle is not None:
        return bundle

    t0 = time.time()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model", "model_cfg", "tokenizer", "audio_cfg"):
        if key not in state:
            raise RuntimeError(
                f"Checkpoint nie zawiera klucza '{key}' — to nie jest checkpoint z train.py"
            )

    tok_meta = state["tokenizer"]
    if tok_meta["type"] == "bpe":
        # checkpoint niesie sciezke wzgledna repo (np. "bpe_pl.model"); CWD apki
        # to aka_asr_demo_app/, wiec wzgledna rozwiazujemy o root repo
        bpe_path = Path(tok_meta["bpe_model_path"])
        if not bpe_path.is_absolute() and not bpe_path.exists():
            bpe_path = Path(__file__).resolve().parent.parent / bpe_path
        tokenizer = SentencePieceTokenizer(str(bpe_path))
    else:
        tokenizer = CharTokenizer(alphabet=tok_meta["alphabet"])

    audio_cfg = AudioConfig(**state["audio_cfg"])
    extractor = LogMelExtractor(audio_cfg)

    model = ConformerTransducer(
        vocab_size=tokenizer.vocab_size,
        n_mels=audio_cfg.n_mels,
        **state["model_cfg"],
    )
    model.load_state_dict(state["model"])
    model.eval().to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[python] model: {n_params:,} par. | epoka {state.get('epoch', '?')} | "
        f"krok {state.get('step', '?')} | best_WER={state.get('best_wer', float('nan')):.2%} | "
        f"device={DEVICE} | zaladowano w {time.time() - t0:.1f}s",
        flush=True,  # embedded interpreter pyo3 nie flushuje stdout przy wyjsciu
    )

    max_symbols = int(state.get("cfg", {}).get("max_symbols_per_step", 3))
    bundle = (model, tokenizer, extractor, max_symbols)
    _cache[checkpoint_path] = bundle
    return bundle


def _features_from_waveform(
    extractor: LogMelExtractor, samples: np.ndarray, sample_rate: int
) -> np.ndarray:
    """
    Bufor [frames, channels] -> log-mel + CMVN, (T, n_mels) float32.
    Odtwarza LogMelExtractor.__call__ dla bufora w pamieci zamiast pliku.
    """
    if samples.ndim == 2:
        wav = samples.mean(axis=1) if samples.shape[1] > 1 else samples[:, 0]
    else:
        wav = samples
    wav_t = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)

    cfg = extractor.cfg
    if sample_rate != cfg.sample_rate:
        resampler = extractor._resamplers.get(sample_rate)
        if resampler is None:
            resampler = AT.Resample(orig_freq=sample_rate, new_freq=cfg.sample_rate)
            extractor._resamplers[sample_rate] = resampler
        wav_t = resampler(wav_t)

    if wav_t.shape[1] < cfg.n_fft:
        raise ValueError(
            f"Nagranie za krotkie: {wav_t.shape[1]} probek po resamplingu " f"(minimum {cfg.n_fft})"
        )

    return cmvn(extractor.features_from_waveform(wav_t))


# --- affect v0 (ASR_DEMO_AFFECT=1): prozodia liczona z audio+tekstu, BEZ modelu ---

_FILLERS = re.compile(r"\b(y{2,}|e{2,}|a{3,}|hm+|mhm|yhm)\b", re.IGNORECASE)


def prosody_features(wav: np.ndarray, sr: int, text: str) -> dict:
    """Sygnaly prozodyczne dla LLM-a (kontrakt v0, bez ML): tempo mowy, pauzy,
    glosnosc, wahania. Energetyczny pseudo-VAD: ramki 30 ms co 10 ms, prog wzgledem
    p95 energii (-25 dB) z podloga absolutna -55 dBFS. Pauzy liczone TYLKO miedzy
    pierwsza a ostatnia ramka mowy (cisza przed/po nagraniu to nie pauza).
    Zwraca dict gotowy do JSON; wartosci zaokraglone (szum pomiarowy < precyzja)."""
    wav = np.asarray(wav, dtype=np.float32)
    dur_s = len(wav) / sr
    frame, hop = int(0.030 * sr), int(0.010 * sr)
    if len(wav) < frame:
        return {"duration_s": round(dur_s, 2)}
    n = 1 + (len(wav) - frame) // hop
    idx = np.arange(n)[:, None] * hop + np.arange(frame)[None, :]
    rms = np.sqrt(np.mean(wav[idx] ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    thr = max(np.percentile(db, 95) - 25.0, -55.0)
    speech = db > thr
    if not speech.any():
        return {"duration_s": round(dur_s, 2), "speech_s": 0.0}
    first, last = np.argmax(speech), len(speech) - 1 - np.argmax(speech[::-1])
    span = speech[first: last + 1]
    speech_s = float(span.sum()) * hop / sr
    # pauzy: ciagi ciszy >= 0.18 s wewnatrz wypowiedzi
    pauses = []
    run = 0
    for s in span:
        if not s:
            run += 1
        else:
            if run * hop / sr >= 0.18:
                pauses.append(run * hop / sr)
            run = 0
    words = len(text.split())
    return {
        "duration_s": round(dur_s, 2),
        "speech_s": round(speech_s, 2),
        "words_per_s": round(words / speech_s, 2) if speech_s > 0 else 0.0,
        "pauses": {
            "count": len(pauses),
            "total_s": round(sum(pauses), 2),
            "longest_s": round(max(pauses), 2) if pauses else 0.0,
        },
        "loudness_dbfs": round(float(db[speech].mean()), 1),
        "dyn_range_db": round(float(np.percentile(db[speech], 95) - np.percentile(db[speech], 5)), 1),
        "hesitations": len(_FILLERS.findall(text)),
    }


_affect_head = None  # (head, enc_dim) ladowane raz z ASR_DEMO_AFFECT_HEAD


def _affect_va(model, feats_t: torch.Tensor, logit_lengths: torch.Tensor) -> dict | None:
    """Affect v1 (opcjonalne): glowica V/A na encoderze ASR. ASR_DEMO_AFFECT_HEAD=
    <sciezka .pt z affect/train_head.py>. Zwraca {"valence","arousal","confidence"}
    albo None (brak/blad -> cicho pomijamy: zla etykieta gorsza niz zadna).
    confidence = val mean CCC glowicy (stala kalibracja v1, nie per-sample)."""
    global _affect_head
    path = os.environ.get("ASR_DEMO_AFFECT_HEAD") or ""
    if not path:
        return None
    try:
        if _affect_head is None:
            from affect.head import AffectHead

            st = torch.load(path, map_location="cpu", weights_only=False)
            head = AffectHead(st["enc_dim"])
            head.load_state_dict(st["head"])
            head.to(DEVICE).eval()
            conf = round(0.5 * (st.get("val_ccc_v", 0) + st.get("val_ccc_a", 0)), 2)
            _affect_head = (head, conf)
            print(f"[python] affect head: {path} (val CCC~{conf}, "
                  f"licencja: {st.get('data_license', '?')})", flush=True)
        head, conf = _affect_head
        Ts = int(logit_lengths[0])
        with torch.inference_mode():
            enc = model.encoder(feats_t)  # (1, T', D); pojedynczy sample -> bez maski
            va = head(enc[:, :Ts])
        return {"valence": round(float(va[0, 0]), 2),
                "arousal": round(float(va[0, 1]), 2), "confidence": conf}
    except Exception as e:  # affect nie moze ubic transkrypcji
        print(f"[python] affect head pominiety ({type(e).__name__}: {e})", flush=True)
        return None


_postproc = None  # PostProcessor budowany raz (laduje fixer), potem cache


def _postprocess(text: str) -> str:
    """Fixer (korekcja + casing + interpunkcja) -> ITN -> reczne regexy. Konfig z
    ASR_DEMO_* env. Ta sama kolejnosc etapow co w ewaluacji (postprocessing.PostProcessor)."""
    global _postproc
    if _postproc is None:
        from postprocessing import PostProcessor

        _postproc = PostProcessor.from_env(device=DEVICE)
        if _postproc.enabled:
            print(
                f"[python] postprocessing: fixer={_postproc.fixer is not None} "
                f"itn={_postproc.itn} rules={_postproc.rules is not None}",
                flush=True,
            )
    if not _postproc.enabled:
        return text
    out = _postproc.process(text)
    if out != text:
        print(f"[python] postproc: {text!r} -> {out!r}", flush=True)
    return out


def transcribe(samples: np.ndarray, sample_rate: int, checkpoint_path: str) -> str:
    """
    Entry point wolany z Rusta.

    Args:
        samples: np.ndarray shape [frames, channels], dtype float32, zakres [-1, 1]
        sample_rate: sample rate nagrania w Hz (natywny dla urzadzenia)
        checkpoint_path: sciezka do checkpointa .pt z train.py

    Returns:
        Transkrypcja jako str.
    """
    samples = np.asarray(samples, dtype=np.float32)
    n_frames = samples.shape[0] if samples.ndim else 0
    print(
        f"[python] audio: shape={tuple(samples.shape)}, sr={sample_rate} Hz, "
        f"{n_frames / max(1, sample_rate):.2f}s",
        flush=True,
    )

    model, tokenizer, extractor, max_symbols = _load_bundle(checkpoint_path)

    feats = _features_from_waveform(extractor, samples, sample_rate)
    feats_t = torch.from_numpy(feats).unsqueeze(0).to(DEVICE)  # (1, T, n_mels)
    logit_lengths = torch.tensor([subsampled_length(feats.shape[0])], dtype=torch.int32)

    t0 = time.time()
    with torch.inference_mode():
        hyp_ids = greedy_decode_batch(
            model, feats_t, logit_lengths, max_symbols_per_step=max_symbols
        )
    text = tokenizer.decode(hyp_ids[0])
    print(
        f"[python] dekodowanie: {time.time() - t0:.2f}s, "
        f"{len(hyp_ids[0])} tokenow, {feats.shape[0]} ramek",
        flush=True,
    )
    # Postprocessing: fixer (korekcja + casing + interpunkcja) -> ITN.
    # Wlaczane przez ASR_DEMO_FIXER / ASR_DEMO_ITN (patrz postprocessing.py).
    final = _postprocess(text)

    # Affect v0 (OPCJONALNE, ASR_DEMO_AFFECT=1): zamiast golego tekstu zwroc JSON
    # {text, text_raw, prosody} -- kontrakt metadanych dla LLM-a ("jak podejsc do
    # uzytkownika"). Prozodia z audio (pseudo-VAD), wahania z SUROWEGO tekstu
    # (postproc moglby je wyczyscic). Rust/PyO3 i tak przenosi string -> zero
    # zmian w kodzie Rust; default (bez flagi) niezmieniony.
    if os.environ.get("ASR_DEMO_AFFECT", "") not in ("", "0"):
        mono = samples.mean(axis=1) if samples.ndim == 2 and samples.shape[1] > 1 else (
            samples[:, 0] if samples.ndim == 2 else samples
        )
        prosody = prosody_features(mono, sample_rate, text)
        print(f"[python] affect: {prosody}", flush=True)
        out = {"text": final, "text_raw": text, "prosody": prosody}
        va = _affect_va(model, feats_t, logit_lengths)  # v1: glowica V/A (opcjonalna)
        if va is not None:
            out["affect"] = va
        return json.dumps(out, ensure_ascii=False)
    return final
