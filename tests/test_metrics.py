"""compute_wer / _edit_distance -- na tej liczbie stoi kazda decyzja (best.pt,
averaging, char vs BPE), wiec bug tutaj = ciche, bledne WER-y."""
import pytest

from train import _edit_distance, compute_wer


def test_edit_distance():
    assert _edit_distance(list("kot"), list("kot")) == 0
    assert _edit_distance(list("kot"), list("kod")) == 1  # substytucja
    assert _edit_distance(list("kot"), list("koty")) == 1  # insercja
    assert _edit_distance(list("kot"), list("ot")) == 1  # delecja
    assert _edit_distance([], list("abc")) == 3
    assert _edit_distance(list("abc"), []) == 3


def test_wer_perfect_and_empty():
    assert compute_wer(["ala ma kota"], ["ala ma kota"]) == 0.0
    assert compute_wer(["ala ma kota"], [""]) == 1.0  # wszystko usuniete


def test_wer_word_level_substitution():
    # 1 blad na 3 slowa
    assert compute_wer(["ala ma kota"], ["ala ma psa"]) == pytest.approx(1 / 3)


def test_wer_micro_average():
    # agregacja: total_dist / total_words = 1 / 5
    wer = compute_wer(["a b", "c d e"], ["a b", "c d x"])
    assert wer == pytest.approx(1 / 5)


def test_wer_insertions_over_100_percent():
    # przeemitowanie (blank-bias na zimno) -> WER moze przekroczyc 100%
    assert compute_wer(["a"], ["a b c d"]) == pytest.approx(3.0)
