"""Affect v1: glowica V/A (pooling maskowany, CCC, mapowanie nEMO). CPU, bez audio."""

import pytest

torch = pytest.importorskip("torch")

from affect.head import EMOTION_VA, AffectHead, AttentivePooling, ccc, ccc_loss  # noqa: E402

NEMO_LABELS = {"anger", "fear", "happiness", "neutral", "sadness", "surprised"}


def test_mapping_pokrywa_etykiety_nemo():
    assert set(EMOTION_VA) == NEMO_LABELS
    for v, a in EMOTION_VA.values():
        assert -1 <= v <= 1 and -1 <= a <= 1


def test_pooling_ignoruje_padding():
    torch.manual_seed(0)
    pool = AttentivePooling(8).eval()
    h = torch.randn(1, 5, 8)
    padded = torch.cat([h, torch.randn(1, 3, 8)], dim=1)  # 3 ramki smieci
    mask = torch.tensor([[False] * 5 + [True] * 3])
    out_clean = pool(h)
    out_masked = pool(padded, mask)
    assert torch.allclose(out_clean, out_masked, atol=1e-6)


def test_ccc_wartosci():
    x = torch.tensor([0.1, 0.5, -0.3, 0.8])
    assert ccc(x, x).item() == pytest.approx(1.0, abs=1e-5)          # idealna zgodnosc
    assert ccc(x, -x).item() < 0                                     # antykorelacja
    loss = ccc_loss(torch.stack([x, x], dim=1), torch.stack([x, x], dim=1))
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_head_forward_zakres_i_ksztalt():
    torch.manual_seed(0)
    head = AffectHead(16).eval()
    enc = torch.randn(3, 7, 16)
    mask = torch.zeros(3, 7, dtype=torch.bool)
    out = head(enc, mask)
    assert out.shape == (3, 2)
    assert out.abs().max() <= 1.0  # tanh
