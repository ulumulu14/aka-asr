"""FAZA 2: cyfry -> polska forma mowiona (spoken_numbers.to_spoken_pl). Wymaga
num2words (requirements-pseudo) -> skip gdy brak, zeby rdzenny suite byl lekki."""

import pytest

pytest.importorskip("num2words")

from utils.spoken_numbers import has_digit, to_spoken_pl


def test_kardynalne():
    assert to_spoken_pl("druk nr 2779") == "druk numer dwa tysiące siedemset siedemdziesiąt dziewięć"


def test_data_dopelniacz():
    # najczestszy wzorzec Sejmu: dzien+miesiac -> dopelniacz porzadkowy + skrot br.
    assert to_spoken_pl("2 lipca br.") == "drugiego lipca bieżącego roku"


def test_skroty():
    assert to_spoken_pl("art. 241") == "artykuł dwieście czterdzieści jeden"
    assert to_spoken_pl("poz. 5") == "pozycja pięć"


def test_porzadkowy_przed_malym_slowem():
    # "61." + male slowo = porzadkowy (mianownik -- przyblizenie)
    assert to_spoken_pl("61. posiedzeniu") == "sześćdziesiąty pierwszy posiedzeniu"


def test_kropka_zdania_to_nie_porzadkowy():
    # kropka przed koncem/wielka litera = kropka zdania, liczba kardynalna
    out = to_spoken_pl("druk 2779. Następny")
    assert "dwa tysiące siedemset siedemdziesiąt dziewięć" in out
    assert "." in out  # kropka zdania zachowana


def test_rok_skrot():
    assert to_spoken_pl("w 2025 r. przyjęto") == "w dwa tysiące dwadzieścia pięć roku przyjęto"


def test_bez_cyfr_bez_zmian():
    assert to_spoken_pl("zwykły tekst bez liczb") == "zwykły tekst bez liczb"


def test_has_digit():
    assert has_digit("art. 5") and not has_digit("bez liczb")
