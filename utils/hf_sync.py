#!/usr/bin/env python3
"""
Synchronizacja eksperymentu na HuggingFace Hub -- metryki (TB + log) co epoke,
best.pt asynchronicznie gdy sie poprawia. Jeden run = podkatalog w repo:
    <repo_id>/conformer_<size>/<timestamp>/{tb/, train.log, best.pt}

Bledy HF (siec, token, limit) NIGDY nie ubijaja treningu -- kazdy push w
try/except z logiem warning. Uploady best.pt ida run_as_future (nie blokuja
petli treningowej); metryki sa lekkie. finalize() domyka pending futures.

Token: HF_TOKEN z env (musi miec prawo ZAPISU -- inny niz read-only do BIGOS).
Wlaczane przez training.hf_repo w configu (brak -> sync wylaczony).
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("asr")


class HfSync:
    """Push metryk i best.pt do prywatnego repo HF. No-op gdy repo_id puste."""

    def __init__(
        self,
        repo_id: str | None,
        run_subdir: str,
        token: str | None = None,
        private: bool = True,
    ):
        self.enabled = bool(repo_id)
        self.repo_id = repo_id
        self.prefix = run_subdir.strip("/")
        self._futures: list = []
        self._best_future = None
        self.api = None
        if not self.enabled:
            return
        try:
            import os

            from huggingface_hub import HfApi

            self.api = HfApi(token=token or os.environ.get("HF_TOKEN"))
            self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
            logger.info(f"HF sync: {repo_id}/{self.prefix} (private={private})")
        except Exception as e:  # brak tokenu/pakietu/sieci -> sync off, trening leci
            logger.warning(f"HF sync wylaczony ({type(e).__name__}: {e})")
            self.enabled = False

    def push_metrics(self, tb_dir: str | Path, log_path: str | Path | None = None) -> None:
        """Lekki upload tb/ (+ train.log) -- wolaj po walidacji epoki."""
        if not self.enabled:
            return
        try:
            tb_dir = Path(tb_dir)
            if tb_dir.is_dir():
                self._futures.append(
                    self.api.upload_folder(
                        folder_path=str(tb_dir),
                        path_in_repo=f"{self.prefix}/tb",
                        repo_id=self.repo_id,
                        commit_message="metryki tb",
                        run_as_future=True,
                    )
                )
            if log_path and Path(log_path).exists():
                self._futures.append(
                    self.api.upload_file(
                        path_or_fileobj=str(log_path),
                        path_in_repo=f"{self.prefix}/train.log",
                        repo_id=self.repo_id,
                        commit_message="log",
                        run_as_future=True,
                    )
                )
        except Exception as e:
            logger.warning(f"HF push metryk nieudany ({type(e).__name__}: {e})")

    def push_best(self, best_path: str | Path) -> None:
        """Async upload best.pt (~1,5 GB) -- wolaj gdy nowy najlepszy WER."""
        if not self.enabled or not Path(best_path).exists():
            return
        try:
            self._best_future = self.api.upload_file(
                path_or_fileobj=str(best_path),
                path_in_repo=f"{self.prefix}/best.pt",
                repo_id=self.repo_id,
                commit_message="nowy best",
                run_as_future=True,
            )
        except Exception as e:
            logger.warning(f"HF push best.pt nieudany ({type(e).__name__}: {e})")

    def finalize(self, timeout: float = 600.0) -> None:
        """Domknij pending uploady na koniec treningu (best jest wazny)."""
        if not self.enabled:
            return
        pending = [f for f in (*self._futures, self._best_future) if f is not None]
        for f in pending:
            try:
                f.result(timeout=timeout)
            except Exception as e:
                logger.warning(f"HF finalize: upload niedokonczony ({type(e).__name__}: {e})")
        logger.info("HF sync: domkniete")
