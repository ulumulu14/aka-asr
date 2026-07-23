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

import os
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
        tokenizer = SentencePieceTokenizer(tok_meta["bpe_model_path"])
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


_fixer_cache: dict[str, object] = {}


def _apply_fixer(model_dir: str, text: str) -> str:
    """Opcjonalna post-korekcja ASR (ASR_DEMO_FIXER). Ta sama klasa AutoFixer co w testach."""
    fx = _fixer_cache.get(model_dir)
    if fx is None:
        from autofixer.fixer import AutoFixer

        fx = AutoFixer(model_dir, device=DEVICE)
        _fixer_cache[model_dir] = fx
        print(f"[python] autofixer zaladowany z {model_dir}", flush=True)
    corrected = fx.correct(text)
    if corrected != text:
        print(f"[python] autofixer: {text!r} -> {corrected!r}", flush=True)
    return corrected


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
    # Postprocessing (kolejnosc wazna): autofixer (korekcja bledow + casing +
    # interpunkcja) NAJPIERW, potem ITN (liczby->cyfry na sformatowanym tekscie).
    fixer_dir = os.environ.get("ASR_DEMO_FIXER")
    if fixer_dir:
        text = _apply_fixer(fixer_dir, text)
    # ITN tylko do wyswietlania; ASR_DEMO_ITN=1 wlacza
    if os.environ.get("ASR_DEMO_ITN", "") not in ("", "0"):
        from itn import itn_pl

        converted = itn_pl(text)
        if converted != text:
            print(f"[python] itn: {text!r} -> {converted!r}", flush=True)
        text = converted
    return text
