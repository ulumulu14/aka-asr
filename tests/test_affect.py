"""Affect v0 (demo): prosody_features -- pseudo-VAD, tempo, pauzy, wahania.
Czysta funkcja na syntetycznym sygnale (bez modelu/mikrofonu). transcriber.py
zyje w aka_asr_demo_app/ (nie-pakiet) -> import przez sys.path."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aka_asr_demo_app"))

torch = pytest.importorskip("torch")  # transcriber importuje torch przy imporcie modulu
from transcriber import _FILLERS, prosody_features  # noqa: E402

SR = 16000
RNG = np.random.default_rng(42)


def _signal(spans, total_s=4.0, amp=0.3):
    """Szum w oknach [start, end) sekund, poza nimi cisza."""
    wav = np.zeros(int(total_s * SR), dtype=np.float32)
    for a, b in spans:
        wav[int(a * SR): int(b * SR)] = RNG.normal(0, amp, int((b - a) * SR)).astype(np.float32)
    return wav


def test_speech_and_pause_detection():
    # 1 s mowy, 0.8 s pauzy, 1 s mowy; cisza przed/po NIE liczy sie jako pauza
    wav = _signal([(0.5, 1.5), (2.3, 3.3)])
    p = prosody_features(wav, SR, "ala ma kota w domu")
    assert 1.6 <= p["speech_s"] <= 2.4
    assert p["pauses"]["count"] == 1
    assert 0.6 <= p["pauses"]["longest_s"] <= 1.0
    assert p["duration_s"] == 4.0
    assert p["words_per_s"] > 0


def test_all_silence():
    p = prosody_features(np.zeros(SR, dtype=np.float32), SR, "")
    assert p.get("speech_s", 0.0) == 0.0


def test_hesitations_from_raw_text():
    wav = _signal([(0.2, 1.2)])
    p = prosody_features(wav, SR, "no yyy to jest eee trudne hmm")
    assert p["hesitations"] == 3


def test_fillers_regex_nie_lapie_slow():
    # 'e'/'y' pojedyncze i zwykle slowa nie sa wahaniami
    assert _FILLERS.findall("ale y e many hymn") == []


def test_too_short_returns_duration_only():
    p = prosody_features(np.zeros(100, dtype=np.float32), SR, "x")
    assert set(p.keys()) == {"duration_s"}
