"""Tokenizery: round-trip (encode->decode zachowuje tekst), blank=0 zarezerwowany.
Gdyby BPE po cichu gubil diakrytyk, byloby to ciche pogorszenie jakosci."""

from pathlib import Path

import pytest

from preprocessing import POLISH_ALPHABET, CharTokenizer, SentencePieceTokenizer

_BPE = Path(__file__).resolve().parents[1] / "bpe_pl.model"
_TEKSTY = ["ala ma kota", "zażółć gęślą jaźń", "dzień dobry państwu"]


def test_char_roundtrip():
    tok = CharTokenizer()
    for s in _TEKSTY:
        assert tok.decode(tok.encode(s)) == s


def test_char_vocab_and_blank():
    tok = CharTokenizer()
    assert tok.vocab_size == len(POLISH_ALPHABET)  # bez blanka
    for s in _TEKSTY:
        assert 0 not in tok.encode(s)  # ID 0 = blank, nigdy nie emitowany


def test_char_drops_out_of_alphabet():
    # znaki spoza alfabetu (cyfry, interpunkcja) sa po cichu wycinane
    # -- dlatego ASRDataset filtruje sample z cyframi PRZED tokenizacja
    tok = CharTokenizer()
    assert tok.decode(tok.encode("a1b!c")) == "abc"


@pytest.mark.skipif(not _BPE.exists(), reason="brak bpe_pl.model")
def test_bpe_roundtrip_and_blank():
    pytest.importorskip("sentencepiece")
    tok = SentencePieceTokenizer(_BPE)
    for s in ["zażółć gęślą jaźń", "dzień dobry państwu"]:
        assert tok.decode(tok.encode(s)) == s  # diakrytyki przezywaja
    # ID sentencepiece przesuniete o +1 -> blank=0 wolny
    assert min(i for s in _TEKSTY for i in tok.encode(s)) >= 1
