"""Smoke pipeline'u treningowego na syntetyku (CPU, bez audio/GPU): model buduje
sie, forward/loss/backward dzialaja, greedy i beam produkuja tokeny w vocab.
Lapie 'zepsulem forward/collate/loss' BEZ odpalania poda."""

import torch

from decoding import beam_search_decode_batch
from model import MODEL_SIZES, ConformerTransducer, rnnt_loss
from preprocessing import subsampled_length
from train import greedy_decode_batch

VOCAB = 10  # bez blanka -> wyjscie jointa ma 11 klas (0=blank, 1..10=tokeny)


def _n_params(size, override=None):
    cfg = dict(MODEL_SIZES[size])
    if override:
        cfg.update(override)
    m = ConformerTransducer(vocab_size=VOCAB, n_mels=80, dropout=0.0, **cfg)
    return sum(p.numel() for p in m.parameters())


def test_sizes_ordered_smol_mid_big_custom():
    # kazdy kolejny rozmiar wiekszy; custom domyslnie > big
    n = {s: _n_params(s) for s in ("smol", "mid", "big", "custom")}
    assert n["smol"] < n["mid"] < n["big"] < n["custom"]


def test_dims_override_scales_model():
    base = _n_params("custom")
    bigger = _n_params("custom", {"enc_dim": 1024, "enc_heads": 16, "joint_dim": 1024})
    assert bigger > base  # override z configu realnie zmienia architekture


def _model():
    torch.manual_seed(0)
    return ConformerTransducer(
        vocab_size=VOCAB, n_mels=80, dropout=0.0, **dict(MODEL_SIZES["smol"])
    )


def test_forward_shape():
    logits = _model().eval()(torch.randn(2, 48, 80), torch.zeros(2, 6, dtype=torch.long))
    assert logits.shape[0] == 2  # batch
    assert logits.shape[2] == 6  # U+1
    assert logits.shape[-1] == VOCAB + 1  # +blank


def test_train_step_backward_finite():
    m = _model().train()
    feats, labels = torch.randn(2, 48, 80), torch.zeros(2, 6, dtype=torch.long)
    logits = m(feats, labels)
    T_sub = logits.shape[1]
    loss = rnnt_loss(
        logits.float(),
        torch.randint(1, VOCAB + 1, (2, 5), dtype=torch.int32),
        torch.tensor([T_sub, T_sub], dtype=torch.int32),
        torch.tensor([5, 5], dtype=torch.int32),
        blank=0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_greedy_and_beam_produce_valid_tokens():
    m = _model().eval()
    feats = torch.randn(2, 48, 80)
    lengths = torch.tensor([subsampled_length(48)] * 2, dtype=torch.int32)
    greedy = greedy_decode_batch(m, feats, lengths)
    beam = beam_search_decode_batch(m, feats, lengths, beam_size=3)
    assert len(greedy) == len(beam) == 2
    # emitowane tokeny to nie-blank z zakresu 1..VOCAB
    assert all(1 <= t <= VOCAB for hyp in greedy + beam for t in hyp)


def test_greedy_return_scores():
    m = _model().eval()
    feats = torch.randn(2, 48, 80)
    lengths = torch.tensor([subsampled_length(48)] * 2, dtype=torch.int32)
    hyps, scores = greedy_decode_batch(m, feats, lengths, return_scores=True)
    assert len(hyps) == len(scores) == 2
    # score = sredni log-prob (<=0) albo -inf dla pustej hipotezy
    assert all(s <= 0 for s in scores)
