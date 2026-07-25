#!/usr/bin/env python3
"""
RuleFixer -- reczne poprawki regex jako OSTATNIE sito postprocessingu.

Do uporczywych bledow, ktorych model (AutoFixer) nie lapie: konkretne nazwy
wlasne, powtarzalne przekrecenia, pisownia laczna. Stosowany "w ostatecznosci"
-- po AutoFixerze i ITN, na finalnym tekscie.

Plik regul (JSON, lista obiektow, stosowane SEKWENCYJNIE w kolejnosci pliku):
    [
      {"pattern": "hugging\\s*face", "replacement": "HuggingFace", "flags": "i",
       "comment": "uporczywy blad ASR"},
      {"pattern": "\\bpy?torcha\\b", "replacement": "PyTorcha"}
    ]

Pola: pattern (regex), replacement (moze miec \\1 backrefs), flags (opcjonalne:
i/m/s/x), comment (opcjonalne, ignorowane). Bledny regex -> jasny ValueError przy
ladowaniu (wskazuje ktora regula).

    from fixer.rules import RuleFixer
    rf = RuleFixer("fixer/rules.json")
    rf.apply("pracuje w hugging face")   # -> "pracuje w HuggingFace"
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_FLAGS = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL, "x": re.VERBOSE}


class RuleFixer:
    """Laduje i kompiluje reguly regex, stosuje je sekwencyjnie do tekstu."""

    def __init__(self, rules_path: str | Path):
        self.rules: list[tuple[re.Pattern, str]] = []
        data = json.loads(Path(rules_path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{rules_path}: oczekiwano listy regul, jest {type(data).__name__}")
        for i, r in enumerate(data):
            flags = 0
            for ch in r.get("flags", ""):
                if ch not in _FLAGS:
                    raise ValueError(f"regula {i}: nieznana flaga '{ch}' (dozwolone: imsx)")
                flags |= _FLAGS[ch]
            try:
                pat = re.compile(r["pattern"], flags)
            except (re.error, KeyError) as e:
                raise ValueError(f"regula {i}: zly wzorzec {r.get('pattern')!r} ({e})") from e
            self.rules.append((pat, r.get("replacement", "")))

    def apply(self, text: str) -> str:
        for pat, repl in self.rules:
            text = pat.sub(repl, text)
        return text

    def __len__(self) -> int:
        return len(self.rules)
