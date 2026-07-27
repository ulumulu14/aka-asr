"""_pick_decode_batches: losowy-deterministyczny podzbior batchy do WER w walidacji.
Naprawia bias 'najkrotsze-N' (dataset sortowany po duracji -> WER mierzyl mikroklipy)."""

from train import _pick_decode_batches


def test_none_when_no_limit():
    assert _pick_decode_batches(100, 10000, None, seed=42) is None


def test_deterministic_same_seed():
    a = _pick_decode_batches(200, 10000, 500, seed=42)
    b = _pick_decode_batches(200, 10000, 500, seed=42)
    assert a == b  # ten sam podzbior co epoke -> stabilna metryka


def test_different_seed_different_subset():
    a = _pick_decode_batches(200, 10000, 500, seed=42)
    b = _pick_decode_batches(200, 10000, 500, seed=7)
    assert a != b


def test_not_a_prefix_spread_over_range():
    # sedno fixa: wybor NIE jest prefiksem (nie same najkrotsze batche)
    picked = _pick_decode_batches(200, 10000, 500, seed=42)
    assert picked != set(range(len(picked)))
    assert max(picked) > 100  # siega w dluzsze partie datasetu


def test_size_covers_wer_max():
    # ~500 sampli przy sredniej 50/batch -> ~10 batchy (+2 marginesu)
    picked = _pick_decode_batches(200, 10000, 500, seed=42)
    assert 10 <= len(picked) <= 12


def test_small_loader_capped():
    picked = _pick_decode_batches(3, 100, 500, seed=42)
    assert picked is not None and len(picked) <= 3
