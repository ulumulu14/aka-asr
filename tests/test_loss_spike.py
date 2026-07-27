"""Logika loss-spike skip: batch z loss_rnnt > multiplier*mediana pomijany.
Sam mechanizm decyzyjny (bez pelnego treningu) -- mediana odporna na outliery."""

import statistics
from collections import deque


def _should_skip(hist, lr_val, mult, warmup=100):
    """Replika warunku z train.py: skip gdy zebrano >=warmup i lr_val > mult*mediana."""
    return len(hist) >= warmup and lr_val > statistics.median(hist) * mult


def test_no_skip_before_warmup():
    hist = deque([50.0] * 50, maxlen=200)  # tylko 50 probek
    assert not _should_skip(hist, 999.0, 3.0)  # <100 -> nie skipuj (buduj statystyke)


def test_skip_spike_after_warmup():
    hist = deque([50.0] * 150, maxlen=200)  # mediana 50
    assert _should_skip(hist, 212.0, 3.0)      # 212 > 150 -> spike, skip
    assert _should_skip(hist, 312.0, 3.0)      # 312 > 150 -> skip


def test_keep_normal_variation():
    hist = deque([50.0] * 150, maxlen=200)
    assert not _should_skip(hist, 77.0, 3.0)   # 77 < 150 -> normalne wahanie, keep
    assert not _should_skip(hist, 149.0, 3.0)  # tuz pod progiem -> keep


def test_median_robust_to_outliers_in_hist():
    # kilka spike'ow w historii nie psuje mediany (w przeciwienstwie do sredniej)
    hist = deque([50.0] * 145 + [300.0] * 5, maxlen=200)
    assert statistics.median(hist) == 50.0       # mediana ignoruje 5 outlierow
    assert _should_skip(hist, 200.0, 3.0)        # prog 150 -> 200 skip


def test_multiplier_zero_semantics():
    # w train.py multiplier 0 wylacza caly blok (nie wchodzi tu); sanity progu
    hist = deque([50.0] * 150, maxlen=200)
    assert _should_skip(hist, 160.0, 3.0)        # 160 > 150
    assert not _should_skip(hist, 160.0, 4.0)    # prog 200 -> 160 keep
