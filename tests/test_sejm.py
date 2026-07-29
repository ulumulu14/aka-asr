"""FAZA 2 Sejm (ground-truth): czyszczenie HTML stenogramu, tokenizacja i AGREEMENT
Whisper<->stenogram (align_segment). Czyste funkcje, bez API/audio/GPU."""

from utils.datasets_conversion.sejm2json import (
    align_segment,
    norm_words,
    strip_committee_html,
    strip_statement_html,
    tokenize_steno,
    word_wer,
    _seg_index,
)


def test_strip_html_usuwa_mowce_didaskalia_tagi():
    raw = ('<h2 class="mowca">Marszałek Szymon Hołownia:</h2>'
           "<p>Otwieram posiedzenie. (Oklaski) Proszę o ciszę.</p>")
    out = strip_statement_html(raw)
    assert "Marszałek" not in out          # naglowek mowcy usuniety
    assert "Oklaski" not in out            # didaskalia w nawiasie usuniete
    assert "<" not in out                  # tagi usuniete
    assert out == "Otwieram posiedzenie. Proszę o ciszę."


def test_strip_html_pelny_dokument_wycina_naglowki():
    # realna struktura API: head+title + h1 (kadencja) + h2 (mowca) + tresc
    raw = ("<html><head><title>62. posiedzenie - Marszałek</title></head><body>"
           '<blockquote><h1>10. kadencja, 62. posiedzenie, 1. dzień (15-07-2026)</h1>'
           '<p class="punkt-tytul">Punkt 1.</p>'
           '<h2 class="mowca">Marszałek:</h2>'
           "<p>Dzień dobry państwu. Otwieram posiedzenie.</p></blockquote></body></html>")
    out = strip_statement_html(raw)
    assert out == "Dzień dobry państwu. Otwieram posiedzenie."
    assert "kadencja" not in out and "posiedzenie, 1" not in out and "Punkt" not in out


def test_tokenize_steno_zachowuje_oryginal():
    orig, norm = tokenize_steno("Wysoka Izbo! Głosujemy.")
    assert orig == ["Wysoka", "Izbo!", "Głosujemy."]   # oryginal z casingiem/interpunkcja
    assert norm == ["wysoka", "izbo", "głosujemy"]      # do dopasowania


def test_norm_words_i_diakrytyki():
    assert norm_words("Zażółć, gęślą JAŹŃ!") == ["zażółć", "gęślą", "jaźń"]


def test_word_wer():
    assert word_wer(["a", "b", "c"], ["a", "b", "c"]) == 0.0
    assert word_wer(["a", "b", "c"], ["a", "x", "c"]) == 1 / 3
    assert word_wer([], []) == 0.0
    assert word_wer(["a"], []) == 1.0


def _steno(text):
    return tokenize_steno(text)


def test_align_trafienie_bierze_span_stenogramu():
    orig, norm = _steno("Panie Marszałku Wysoka Izbo dziękuję bardzo za głos")
    # Whisper prawie trafil (jeden blad) -> span stenogramu, niski WER, kursor do przodu
    hyp = norm_words("wysoka izbo dziekuje bardzo")
    res, new_cursor = align_segment(hyp, norm, orig, cursor=0)
    assert res is not None
    assert res["orig"][0] == "Wysoka" and res["orig"][-1] == "bardzo"
    assert res["wer"] <= 0.3
    assert new_cursor > 0  # kotwica przesunieta za dopasowany fragment


def test_align_brak_dopasowania_nie_rusza_kursora():
    orig, norm = _steno("Panie Marszałku Wysoka Izbo głosujemy nad ustawą")
    res, new_cursor = align_segment(norm_words("zupełnie inny tekst xyz qwerty"), norm, orig, cursor=0)
    # difflib nie znajdzie wspolnych slow -> None, kursor bez zmian
    assert res is None and new_cursor == 0


def test_align_monotoniczny_postep():
    # dwa segmenty po kolei -> kursor rosnie, drugi dopasowany DALEJ w stenogramie
    orig, norm = _steno("alfa beta gamma delta epsilon dzeta eta theta")
    r1, c1 = align_segment(norm_words("alfa beta gamma"), norm, orig, cursor=0)
    r2, c2 = align_segment(norm_words("epsilon dzeta eta"), norm, orig, cursor=c1)
    assert r1["orig"][0] == "alfa"
    assert r2["orig"][0] == "epsilon"
    assert c2 > c1


def test_strip_committee_html():
    # realna struktura zapisu komisji: naglowek z boldami BEZ dwukropka (obecnosc),
    # potem wypowiedzi <p><b>Mowca (Partia):</b><br>tekst
    raw = ("<html><body>Porzadek: <b>– pierwsze czytanie</b> ustawy."
           "W posiedzeniu udzial wzieli: <b>Jan Kowalski</b>&nbsp;minister."
           "<p><b>Przewodnicząca poseł Anna Nowak (KO):</b><br>"
           "Dzień dobry państwu. (Gwar na sali) Otwieram posiedzenie."
           "<p><b>Poseł Jan Kowalski (PiS):</b><br>Mam pytanie do ministra.</body></html>")
    out = strip_committee_html(raw)
    assert out == "Dzień dobry państwu. Otwieram posiedzenie. Mam pytanie do ministra."
    assert "Nowak" not in out and "czytanie" not in out and "Gwar" not in out


def test_seg_index_przywraca_kolejnosc_czasowa():
    paths = ["/a/vid_0002.wav", "/a/vid_0000.wav", "/a/vid_0010.wav", "/a/vid_0001.wav"]
    assert sorted(paths, key=_seg_index) == [
        "/a/vid_0000.wav", "/a/vid_0001.wav", "/a/vid_0002.wav", "/a/vid_0010.wav",
    ]
