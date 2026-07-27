#!/usr/bin/env python3
"""
ITN (inverse text normalization) dla polskiego: forma mowiona -> pisana.

Zakres V1: liczebniki glowne w mianowniku ("dwa tysiace pietnascie" -> "2015").
Anty-nadgorliwosc: pojedyncze slowo o wartosci < 10 zostaje slowem ("jeden z nich"
bez zmian); konwertowane sa frazy >= 2 slow liczbowych albo o wartosci >= 10.
Formy przypadkow zaleznych (dwoch, pieciu, stu, dwustu...) NIE sa konwertowane.

To warstwa WYSWIETLANIA (demo, UI). Nie stosowac przy liczeniu WER, bo
referencje sa w formie mowionej (dataset odrzuca transkrypcje z cyframi).

Uzycie:  from itn import itn_pl;  itn_pl("dwa tysiące piętnaście")  -> "2015"
CLI:     python itn.py "tekst..."     (bez argumentu czyta stdin)
"""

from __future__ import annotations

import sys

UNITS = {
    "jeden": 1,
    "jedna": 1,
    "jedno": 1,
    "dwa": 2,
    "dwie": 2,
    "trzy": 3,
    "cztery": 4,
    "pięć": 5,
    "sześć": 6,
    "siedem": 7,
    "osiem": 8,
    "dziewięć": 9,
}
TEENS = {
    "dziesięć": 10,
    "jedenaście": 11,
    "dwanaście": 12,
    "trzynaście": 13,
    "czternaście": 14,
    "piętnaście": 15,
    "szesnaście": 16,
    "siedemnaście": 17,
    "osiemnaście": 18,
    "dziewiętnaście": 19,
}
TENS = {
    "dwadzieścia": 20,
    "trzydzieści": 30,
    "czterdzieści": 40,
    "pięćdziesiąt": 50,
    "sześćdziesiąt": 60,
    "siedemdziesiąt": 70,
    "osiemdziesiąt": 80,
    "dziewięćdziesiąt": 90,
}
HUNDREDS = {
    "sto": 100,
    "dwieście": 200,
    "trzysta": 300,
    "czterysta": 400,
    "pięćset": 500,
    "sześćset": 600,
    "siedemset": 700,
    "osiemset": 800,
    "dziewięćset": 900,
}
# wartosc, czy_forma_pojedyncza (pojedyncza moze wystapic bez licznika: "tysiac zlotych")
SCALES = {
    "tysiąc": (1_000, True),
    "tysiące": (1_000, False),
    "tysięcy": (1_000, False),
    "milion": (1_000_000, True),
    "miliony": (1_000_000, False),
    "milionów": (1_000_000, False),
    "miliard": (1_000_000_000, True),
    "miliardy": (1_000_000_000, False),
    "miliardów": (1_000_000_000, False),
}


def _parse_number(words: list[str], start: int) -> tuple[int, int] | None:
    """
    Probuje sparsowac liczebnik od words[start]. Zwraca (wartosc, ile_slow) albo None.
    Grupa < 1000 musi byc malejaca (setki -> dziesiatki/nastki -> jednosci),
    skale malejace (tysiace po milionach); naruszenie konczy fraze.
    """
    total = 0
    group = 0
    stage = 0  # 0=start grupy, 1=po setkach, 2=po dziesiatkach, 3=po nastkach/jednosciach
    last_scale: int | None = None
    j = start

    while j < len(words):
        w = words[j]
        if w == "zero" and j == start:
            group, stage = 0, 3
        elif w in HUNDREDS and stage == 0:
            group += HUNDREDS[w]
            stage = 1
        elif w in TEENS and stage <= 1:
            group += TEENS[w]
            stage = 3
        elif w in TENS and stage <= 1:
            group += TENS[w]
            stage = 2
        elif w in UNITS and stage <= 2:
            group += UNITS[w]
            stage = 3
        elif w in SCALES:
            value, singular = SCALES[w]
            if group == 0 and not singular:
                break  # "tysiace ludzi" bez licznika -- nie liczba
            if last_scale is not None and value >= last_scale:
                break  # skale musza malec
            total += (group if group > 0 else 1) * value
            group, stage, last_scale = 0, 0, value
        else:
            break
        j += 1

    n = j - start
    if n == 0:
        return None
    return total + group, n


def itn_pl(text: str) -> str:
    """Zamienia frazy liczebnikowe na cyfry; reszte tekstu zostawia bez zmian."""
    words = text.split()
    out: list[str] = []
    i = 0

    while i < len(words):
        parsed = _parse_number(words, i)

        if parsed is not None:
            value, n = parsed

            if n >= 2 or value >= 10:
                out.append(str(value))
                i += n
                continue

        out.append(words[i])
        i += 1

    return " ".join(out)


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else sys.stdin.read()
    print(itn_pl(text.strip()))
