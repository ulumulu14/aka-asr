"""Metryki held-out ewaluacji (test.py): CER, rozpoznanie zrodla, _acoustic.
test.py ladowany przez importlib -- 'test' jako nazwa modulu kolidowalaby ze stdlib."""

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "heldout_eval", Path(__file__).resolve().parents[1] / "test.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_acoustic, _source, compute_cer = _mod._acoustic, _mod._source, _mod.compute_cer


def test_source_from_path():
    assert _source("/w/data/cv-corpus-26/wav/x.wav") == "cv"
    assert _source("/w/data/mls_polish/train/a.flac") == "mls"
    assert _source("/w/data/bigos/pjatk/x.wav") == "bigos"
    assert _source("/w/data/voxpopuli_pl/train/x.wav") == "voxpopuli"
    assert _source("/w/data/yodas_pl/0/x.wav") == "yodas"
    assert _source("/w/data/cos_innego/x.wav") == "inne"


def test_cer_char_level():
    assert compute_cer(["kot"], ["kot"]) == 0.0
    assert compute_cer(["kot"], ["kat"]) == 1 / 3
    assert compute_cer(["abc"], [""]) == 1.0
    assert compute_cer(["ala ma kota"], ["ala ma koty"]) < 1 / 3


def test_acoustic_normalizes_for_fair_wer():
    assert _acoustic("Ala, MA kota!") == "ala ma kota"
    assert _acoustic("Dwa Tysiace.") == "dwa tysiace"
