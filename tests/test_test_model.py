"""Metryki held-out ewaluacji (utils/test_model): CER, rozpoznanie zrodla, _acoustic."""

from utils.test_model import _acoustic, _source, compute_cer


def test_source_from_path():
    assert _source("/w/data/cv-corpus-26/wav/x.wav") == "cv"
    assert _source("/w/data/mls_polish/train/a.flac") == "mls"
    assert _source("/w/data/bigos/pjatk/x.wav") == "bigos"
    assert _source("/w/data/voxpopuli_pl/train/x.wav") == "voxpopuli"
    assert _source("/w/data/yodas_pl/0/x.wav") == "yodas"
    assert _source("/w/data/cos_innego/x.wav") == "inne"


def test_cer_char_level():
    assert compute_cer(["kot"], ["kot"]) == 0.0
    assert compute_cer(["kot"], ["kat"]) == 1 / 3          # 1 znak na 3
    assert compute_cer(["abc"], [""]) == 1.0               # wszystko usuniete
    # CER < WER: literowka psuje 1 slowo (WER) ale malo znakow (CER)
    assert compute_cer(["ala ma kota"], ["ala ma koty"]) < 1 / 3


def test_acoustic_normalizes_for_fair_wer():
    # casing/interpunkcja usuniete -> WER/CER porownywalne char vs BPE
    assert _acoustic("Ala, MA kota!") == "ala ma kota"
    assert _acoustic("Dwa Tysiace.") == "dwa tysiace"
