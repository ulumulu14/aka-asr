"""RuleFixer -- reczne poprawki regex (ostatnie sito postprocessingu): ladowanie,
kolejnosc, flagi, backreferencje, jasny blad na zlym regexie. Bez zaleznosci."""

import json

import pytest

from fixer.rules import RuleFixer


def _write(tmp_path, rules):
    p = tmp_path / "rules.json"
    p.write_text(json.dumps(rules), encoding="utf-8")
    return p


def test_basic_replace_and_flags(tmp_path):
    rf = RuleFixer(_write(tmp_path, [
        {"pattern": "hugging\\s*face", "replacement": "HuggingFace", "flags": "i"},
    ]))
    assert len(rf) == 1
    assert rf.apply("pracuje w hugging face") == "pracuje w HuggingFace"
    assert rf.apply("HUGGINGFACE rules") == "HuggingFace rules"  # flaga i


def test_sequential_order(tmp_path):
    # reguly stosowane po kolei: 2. widzi wynik 1.
    rf = RuleFixer(_write(tmp_path, [
        {"pattern": "kot", "replacement": "pies"},
        {"pattern": "pies", "replacement": "ptak"},
    ]))
    assert rf.apply("kot") == "ptak"


def test_backreference(tmp_path):
    rf = RuleFixer(_write(tmp_path, [
        {"pattern": "\\bpaj(torch\\w*)\\b", "replacement": "Py\\1", "flags": "i"},
    ]))
    assert rf.apply("uzywam pajtorcha") == "uzywam Pytorcha"


def test_no_match_passthrough(tmp_path):
    rf = RuleFixer(_write(tmp_path, [{"pattern": "xyz", "replacement": "abc"}]))
    assert rf.apply("nic tu nie pasuje") == "nic tu nie pasuje"


def test_bad_regex_raises_with_index(tmp_path):
    with pytest.raises(ValueError, match="regula 1"):
        RuleFixer(_write(tmp_path, [
            {"pattern": "ok", "replacement": "x"},
            {"pattern": "(niezamkniety", "replacement": "y"},
        ]))


def test_unknown_flag_raises(tmp_path):
    with pytest.raises(ValueError, match="nieznana flaga"):
        RuleFixer(_write(tmp_path, [{"pattern": "a", "replacement": "b", "flags": "z"}]))
