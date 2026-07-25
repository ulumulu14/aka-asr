#!/usr/bin/env python3
"""
Postprocessing tekstu ASR -- jeden pipeline dla inferencji i ewaluacji end-to-end.

Etapy (kolejnosc wazna):
  1. AutoFixer (plT5): korekcja bledow ASR + przywrocenie CASINGU (wielkie litery,
     nazwy wlasne "HuggingFace") + interpunkcji. Jeden model seq2seq robi to razem
     (patrz autofixer/).
  2. ITN (itn.py): liczby slowami -> cyfry, na juz sformatowanym tekscie.

Trzymanie kolejnosci w JEDNYM miejscu (zamiast powielac hooki w transcriber /
evaluate / batch-inference) chroni przed rozjazdem inferencja<->ewaluacja -- ta
sama zasada co wspolna klasa AutoFixer dla testu i inferencji.

transformers/itn importowane leniwie: samo `import postprocessing` nie ciagnie
ciezkich zaleznosci ani nie wymaga fixera, gdy zaden etap nie jest wlaczony.
"""

from __future__ import annotations

import os


class PostProcessor:
    """Sklada wlaczone etapy postprocessingu. Bez zadnego = no-op (tani passthrough)."""

    def __init__(
        self,
        fixer_dir: str | None = None,
        itn: bool = False,
        device: str | None = None,
    ):
        self.fixer = None
        if fixer_dir:
            from autofixer.fixer import AutoFixer  # lazy: transformers ciezkie

            self.fixer = AutoFixer(fixer_dir, device=device)
        self.itn = itn
        self._itn_fn = None
        if itn:
            from itn import itn_pl

            self._itn_fn = itn_pl

    @property
    def enabled(self) -> bool:
        return self.fixer is not None or self.itn

    def process(self, text: str) -> str:
        return self.process_batch([text])[0]

    def process_batch(self, texts: list[str]) -> list[str]:
        out = list(texts)
        if self.fixer is not None:
            out = self.fixer.correct_batch(out)
        if self._itn_fn is not None:
            out = [self._itn_fn(t) for t in out]
        return out

    @classmethod
    def from_env(cls, device: str | None = None) -> "PostProcessor":
        """Z ASR_DEMO_FIXER (sciezka modelu) i ASR_DEMO_ITN (1/0) -- dla demo/transcriber."""
        return cls(
            fixer_dir=os.environ.get("ASR_DEMO_FIXER") or None,
            itn=os.environ.get("ASR_DEMO_ITN", "") not in ("", "0"),
            device=device,
        )
