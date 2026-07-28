"""FAZA 2: pakowanie spanow VAD w segmenty treningowe (longform2json.pack_spans)
i filtry jakosci pseudo-etykiet (pseudo_whisper.quality_check). Czyste funkcje,
bez audio/GPU/transformers (importy ciezkie sa lazy w main())."""

from utils.datasets_conversion.longform2json import _license_ok, pack_spans
from utils.pseudo_whisper import quality_check


def test_pack_merges_close_spans():
    # frazy przedzielone krotka pauza (<=merge_gap) sklejaja sie w jedno zdanie
    out = pack_spans([(0.0, 2.0), (2.3, 5.0)], min_seg=3.0, max_seg=18.0, merge_gap=0.5)
    assert out == [(0.0, 5.0)]


def test_pack_no_merge_over_gap():
    # dluga pauza rozdziela; izolowany kawalek < min_seg odpada
    out = pack_spans([(0.0, 4.0), (10.0, 12.0)], min_seg=3.0, max_seg=18.0, merge_gap=0.5)
    assert out == [(0.0, 4.0)]


def test_pack_splits_long_span_evenly():
    # 40 s mowy -> 3 rowne kawalki ~13.3 s, wszystkie w [min, max]
    out = pack_spans([(0.0, 40.0)], min_seg=3.0, max_seg=18.0)
    assert len(out) == 3
    for a, b in out:
        assert 3.0 <= b - a <= 18.0
    assert abs(out[-1][1] - 40.0) < 1e-6  # pokrywa calosc, bez mikro-ogona


def test_pack_merge_respects_max():
    # scalanie nie przekracza max_seg -- drugi span zostaje osobno
    out = pack_spans([(0.0, 17.0), (17.2, 30.0)], min_seg=3.0, max_seg=18.0, merge_gap=0.5)
    assert out[0] == (0.0, 17.0)
    assert len(out) == 2


def test_pack_empty():
    assert pack_spans([]) == []


def test_quality_ok_zdrowa_polszczyzna():
    ok, reason = quality_check("To jest zwykle polskie zdanie o niczym.", duration=3.0)
    assert ok, reason


def test_quality_drops_empty_and_watermark():
    assert quality_check("", 3.0)[1] == "pusty"
    assert quality_check("Napisy stworzone przez spolecznosc Amara.org", 5.0)[1] == "watermark"


def test_quality_drops_digits_unless_kept():
    assert quality_check("mam 2 koty", 3.0)[1] == "cyfry"
    assert quality_check("mam 2 koty", 3.0, keep_digits=True)[0]


def test_quality_cps_bounds():
    # 4 znaki w 10 s = halucynacja na ciszy; 300 znakow w 3 s = smieci
    assert quality_check("tak.", 10.0)[1] == "cps_niski"
    assert quality_check("a" * 300, 3.0)[1] == "cps_wysoki"


def test_quality_drops_decoder_loop():
    text = " ".join(["tak jest"] * 12)  # petla dekodera: te same trigramy w kolko
    assert quality_check(text, 8.0)[1] == "powtorzenia"


def test_license_gate_off_when_not_required():
    # brak wymogu (Tier 1: Sejm/wlasne audio) -> wszystko przechodzi
    assert _license_ok({"license": "Standard YouTube License"}, None)
    assert _license_ok({}, None)


def test_license_gate_accepts_cc_by():
    prov = {"license": "Creative Commons Attribution license (reuse allowed)"}
    assert _license_ok(prov, "Creative Commons")


def test_license_gate_rejects_standard_and_missing():
    # fail-safe: standardowa licencja YT ORAZ brak info.json -> odrzut przy wymogu CC
    assert not _license_ok({"license": "Standard YouTube License"}, "Creative Commons")
    assert not _license_ok({}, "Creative Commons")
