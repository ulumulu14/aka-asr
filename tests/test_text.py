"""normalize_text + filtr cyfr + ITN. Bug w normalizacji = cicha korupcja
danych treningowych (patrz saga z '_' w VoxPopuli)."""

from itn import itn_pl
from preprocessing import _DIGIT_RE, normalize_text


def test_normalize_lowercase_whitespace_dash():
    assert normalize_text("Ala  MA\tkota") == "ala ma kota"
    assert normalize_text("dzień - dobry") == "dzień dobry"  # myslnik -> spacja
    assert normalize_text("  x  y  ") == "x y"


def test_normalize_zachowuje_diakrytyki():
    # regresja na sage terminalowa: normalizacja NIE tyka polskich znakow
    assert normalize_text("Zażółć Gęślą JAŹŃ") == "zażółć gęślą jaźń"


def test_digit_filter():
    assert _DIGIT_RE.search("mam 5 kotów")
    assert not _DIGIT_RE.search("mam pięć kotów")


def test_itn_cardinals():
    cases = [
        ("dwa tysiące piętnaście", "2015"),
        ("sto dwadzieścia trzy", "123"),
        ("dziesięć osób", "10 osób"),
        ("tysiąc dziewięćset osiemdziesiąt dziewięć", "1989"),
        ("dwa miliony pięćset tysięcy", "2500000"),
    ]
    for src, want in cases:
        assert itn_pl(src) == want


def test_itn_anty_nadgorliwosc():
    # pojedyncze slowo o wartosci < 10 zostaje slowem
    assert itn_pl("jeden z nich poszedł") == "jeden z nich poszedł"
    assert itn_pl("pięć kotów") == "pięć kotów"
    # mnoga skala bez licznika to nie liczba
    assert itn_pl("tysiące ludzi") == "tysiące ludzi"
