"""Repo root na sys.path, zeby testy importowaly moduly z korzenia (train, model, ...)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
