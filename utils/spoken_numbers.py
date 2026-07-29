#!/usr/bin/env python3
"""
Cyfry -> polska forma MOWIONA (spoken form) dla etykiet ASR. Whisper i stenogramy
Sejmu pisza liczby cyframi ("druk nr 2779", "2 lipca", "61. posiedzeniu"), a nasz
trening operuje na SLOWACH (ITN dopiero na wyjsciu) i filtr tnie sample z cyframi.
Konwersja zachowuje wolumen (zamiast dropowac) i przyblizza to, co realnie padlo.

Reguly kontekstowe (num2words nie odmienia przez przypadki -> lekki szum
na koncowkach, swiadomie zaakceptowany):
  * skroty: nr->numer, art.->artykul, ust.->ustep, pkt->punkt, poz.->pozycja,
    br.->biezacego roku, tj.->to jest
  * daty "<dzien> <miesiac>" -> dzien w DOPELNIACZU porzadkowym ("2 lipca"->
    "drugiego lipca") -- najczestszy wzorzec, wiec warto go zrobic dobrze
  * "<n>." przed malym slowem = porzadkowy ("61. posiedzeniu"->"szescdziesiaty
    pierwszy...") -- mianownik (przyblizenie); kropka przed WIELKA/koncem = kropka
    zdania, liczba kardynalna
  * reszta -> kardynalne

Wymaga num2words (requirements-pseudo.txt). Import leniwy -> `import spoken_numbers`
nie ciagnie zaleznosci, dopoki nie wolasz to_spoken_pl.
"""
import re

_MONTHS = ("stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca", "lipca",
           "sierpnia", "września", "października", "listopada", "grudnia")
_MONTHS_RE = "|".join(_MONTHS)

# dni miesiaca w dopelniaczu porzadkowym (data: "2 lipca" -> "drugiego lipca")
_DAY_GEN = {
    1: "pierwszego", 2: "drugiego", 3: "trzeciego", 4: "czwartego", 5: "piątego",
    6: "szóstego", 7: "siódmego", 8: "ósmego", 9: "dziewiątego", 10: "dziesiątego",
    11: "jedenastego", 12: "dwunastego", 13: "trzynastego", 14: "czternastego",
    15: "piętnastego", 16: "szesnastego", 17: "siedemnastego", 18: "osiemnastego",
    19: "dziewiętnastego", 20: "dwudziestego", 21: "dwudziestego pierwszego",
    22: "dwudziestego drugiego", 23: "dwudziestego trzeciego", 24: "dwudziestego czwartego",
    25: "dwudziestego piątego", 26: "dwudziestego szóstego", 27: "dwudziestego siódmego",
    28: "dwudziestego ósmego", 29: "dwudziestego dziewiątego", 30: "trzydziestego",
    31: "trzydziestego pierwszego",
}


def to_spoken_pl(text: str) -> str:
    """Zamien cyfry i skroty na polska forme mowiona (patrz docstring modulu)."""
    from num2words import num2words

    card = lambda n: num2words(n, lang="pl")
    # 1. skroty (jawne wzorce, zeby nie trafic w srodek slowa jak 'artykul')
    text = re.sub(r"\bnr\b\.?", "numer", text)
    text = re.sub(r"\bart\.", "artykuł", text)
    text = re.sub(r"\bust\.", "ustęp", text)
    text = re.sub(r"\bpkt\b\.?", "punkt", text)
    text = re.sub(r"\bpoz\.", "pozycja", text)
    text = re.sub(r"\bbr\.", "bieżącego roku", text)
    text = re.sub(r"\br\.(?!\w)", "roku", text)  # "2025 r." -> "... roku"
    text = re.sub(r"\btj\.", "to jest", text)
    # 2. daty: dzien + miesiac -> dopelniacz porzadkowy
    text = re.sub(
        r"\b(\d{1,2})\s+(" + _MONTHS_RE + r")\b",
        lambda m: f"{_DAY_GEN.get(int(m.group(1)), card(int(m.group(1))))} {m.group(2)}",
        text,
    )
    # 3. porzadkowy: <n>. + male slowo (kontynuacja zdania) -> ordinal (mianownik)
    text = re.sub(
        r"\b(\d+)\.(?=\s+[a-ząćęłńóśźż])",
        lambda m: num2words(int(m.group(1)), lang="pl", to="ordinal"),
        text,
    )
    # 4. reszta liczb -> kardynalne (kropka zdania zostaje)
    text = re.sub(r"\b\d+\b", lambda m: card(int(m.group())), text)
    return re.sub(r"\s+", " ", text).strip()


def has_digit(text: str) -> bool:
    return any(c.isdigit() for c in text)


if __name__ == "__main__":
    for s in ("druk nr 2779", "W dniu 2 lipca br. zmarła", "Na 61. posiedzeniu Sejm",
              "art. 241 regulaminu", "punkt 15. porządku"):
        print(f"{s!r:45} -> {to_spoken_pl(s)!r}")
