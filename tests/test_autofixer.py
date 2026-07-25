"""Metryki ewaluacji autofixera: WER akustyczny (korekcja), interpunkcja F1, casing
recall. Czyste funkcje (bez transformers) -- rozdzielaja 3 osie jakosci."""

from autofixer.evaluate import _acoustic, _cap_recall, _punct_f1


def test_acoustic_strips_case_and_punct():
    # do WER korekcji: lowercase + bez interpunkcji, diakrytyki zostaja
    assert _acoustic("Dwa Tysiące, piętnaście!") == "dwa tysiące piętnaście"
    assert _acoustic("Zażółć GĘŚLĄ jaźń.") == "zażółć gęślą jaźń"


def test_punct_f1_perfect_and_missing():
    # identyczna interpunkcja -> F1 1
    assert _punct_f1(["ala, ma kota."], ["ala, ma kota."])[2] == 1.0
    # hyp bez interpunkcji -> recall 0
    p, r, f1 = _punct_f1(["ala, ma kota."], ["ala ma kota"])
    assert r == 0.0 and f1 == 0.0
    # hyp odzyskal interpunkcje -> wysoki recall
    assert _punct_f1(["ala, ma kota."], ["ala, ma kota."])[1] == 1.0


def test_cap_recall():
    # 2 slowa z wielka litera w ref (Ala, Kota); hyp lowercase -> 0/2
    acc, total = _cap_recall(["Ala ma Kota"], ["ala ma kota"])
    assert total == 2 and acc == 0.0
    # hyp trafil casing -> 2/2
    acc2, _ = _cap_recall(["Ala ma Kota"], ["Ala ma Kota"])
    assert acc2 == 1.0
    # ref bez casingu -> total 0 (nan acc, nie liczymy)
    _, total0 = _cap_recall(["ala ma kota"], ["ala ma kota"])
    assert total0 == 0
