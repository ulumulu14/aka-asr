#!/usr/bin/env python3
"""
AutoFixer -- wspolny etap post-korekcji tekstu ASR (fine-tune plT5).

Uzywany W OBU sciezkach pipeline'u:
  - testowanie: fixer/evaluate.py (WER ASR-samego vs ASR+fixer),
  - inferencja: asr_demo_app/transcriber.py (korekta transkrypcji na zywo).
Ta sama klasa i te same wagi w obu miejscach -> WER z testu odpowiada temu, co
realnie dostaje uzytkownik.

transformers/torch importowane leniwie w __init__, zeby samo `import fixer`
nie wymagalo ciezkich zaleznosci (wazne np. dla srodowiska treningu ASR).
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PREFIX = "popraw: "  # prefiks zadania T5 (musi byc taki sam w treningu i inferencji)


class AutoFixer:
    """Laduje fine-tunowany plT5 i poprawia tekst ASR. Prefiks/max_len z fixer_meta.json."""

    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        max_len: int = 128,
        num_beams: int = 1,
        guard_max_len_ratio: float | None = None,
    ):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_dir = Path(model_dir)
        meta = {}
        meta_path = model_dir / "fixer_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.prefix = meta.get("prefix", DEFAULT_PREFIX)
        self.max_len = int(meta.get("max_len", max_len))
        self.num_beams = num_beams
        # guard anty-halucynacyjny: odrzuc korekte, ktora zmienia dlugosc > ratio x
        self.guard_max_len_ratio = guard_max_len_ratio

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir)).eval().to(self.device)

    def correct_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        torch = self._torch
        inputs = [self.prefix + (t or "") for t in texts]
        enc = self.tok(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**enc, max_length=self.max_len, num_beams=self.num_beams)
        fixed = [f.strip() for f in self.tok.batch_decode(out, skip_special_tokens=True)]
        if self.guard_max_len_ratio is not None:
            fixed = [self._guard(orig, f) for orig, f in zip(texts, fixed)]
        return fixed

    def correct(self, text: str) -> str:
        return self.correct_batch([text])[0]

    def _guard(self, original: str, fixed: str) -> str:
        """Zostaw oryginal, jesli korekta drastycznie zmienia dlugosc (halucynacja)."""
        ratio = len(fixed) / max(1, len(original))
        r = self.guard_max_len_ratio
        if ratio > r or ratio < 1.0 / r:
            return original
        return fixed
