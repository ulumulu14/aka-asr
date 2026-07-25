"""Deterministyczna matematyka front-endu: subsampling, CMVN, speed perturbation.
Lapie 'przesunalem o ramke' / zle znormalizowalem cechy."""

import json

import numpy as np
import torch

from preprocessing import (
    ASRDataset,
    CharTokenizer,
    cmvn,
    speed_perturb_waveform,
    subsampled_length,
)


def _tiny_manifest(tmp_path, n=5):
    m = tmp_path / "m.jsonl"
    rows = [{"audio_path": f"/x/{i}.wav", "duration": 2.0, "text": "ala ma kota"} for i in range(n)]
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(m)


def test_copies_duplicates_only_with_augment(tmp_path):
    m = _tiny_manifest(tmp_path, n=5)
    tok = CharTokenizer()
    assert len(ASRDataset(m, tok, augment=True, copies=1)) == 5
    dup = ASRDataset(m, tok, augment=True, copies=3)
    assert len(dup) == 15 and len(dup.durations) == 15  # 3x kroki/epoke
    # val (augment=False) NIE dubluje -- kopie tylko dla treningu
    assert len(ASRDataset(m, tok, augment=False, copies=3)) == 5


def test_subsampled_length_divides_by_4_monotonic():
    assert subsampled_length(40) == 10
    assert subsampled_length(400) == 100
    lengths = [subsampled_length(t) for t in range(1, 500)]
    assert all(b >= a for a, b in zip(lengths, lengths[1:]))  # niemalejaca


def test_cmvn_zero_mean_unit_var_per_feature():
    x = (np.random.randn(200, 80) * 5 + 3).astype("float32")
    y = cmvn(x)
    assert np.allclose(y.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(y.std(axis=0), 1, atol=1e-2)


def test_speed_perturb_length():
    wav = torch.randn(1, 16000)
    assert speed_perturb_waveform(wav, 16000, 1.0).shape[1] == 16000  # no-op
    slower = speed_perturb_waveform(wav, 16000, 0.9).shape[1]
    faster = speed_perturb_waveform(wav, 16000, 1.1).shape[1]
    assert slower > 16000 > faster  # 0.9x = dluzej, 1.1x = krocej
    assert abs(slower - 16000 / 0.9) < 5
