#!/usr/bin/env python3
"""
Affect v1: glowica walencja/pobudzenie (V/A) na ZAMROZONYM encoderze ASR.

Architektura: attentive pooling po czasie (maskowany softmax) -> LayerNorm ->
MLP -> 2 wyjscia tanh [-1,1] (valence, arousal). ~1-2M param; encoder liczy sie
raz (wspolny z ASR) -> affect w inferencji ~za darmo. Trening TYLKO glowicy
(encoder frozen/eval) -> zero ryzyka regresji ASR.

Taksonomia (decyzja z brainstormu): wymiary Russella zamiast kategorii; mapowanie
kategorii nEMO na prototypy V/A ponizej (przyblizenie literaturowe -- akceptowany
szum, target i tak jest ciagly). Loss: 1-CCC (concordance), standard w dimensional SER.

UWAGA LICENCJA: wagi trenowane na nEMO (CC-BY-NC-SA) = PROTOTYP niekomercyjny;
kod jest czysty -- komercyjne wagi wymagaja czystego zrodla (patrz pamiec projektu).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# prototypy V/A dla kategorii nEMO (circumplex Russella; przyblizenie)
EMOTION_VA = {
    "neutral":   (0.0,  0.0),
    "happiness": (0.7,  0.5),
    "surprised": (0.2,  0.7),
    "anger":     (-0.6, 0.7),
    "fear":      (-0.6, 0.6),
    "sadness":   (-0.6, -0.4),
}


class AttentivePooling(nn.Module):
    """Wazona suma po czasie: w = softmax(v^T tanh(W h)); padding maskowany -inf."""

    def __init__(self, d_model: int, d_attn: int = 128):
        super().__init__()
        self.proj = nn.Linear(d_model, d_attn)
        self.score = nn.Linear(d_attn, 1, bias=False)

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        # h: (B, T, D); key_padding_mask: (B, T) True = padding
        s = self.score(torch.tanh(self.proj(h))).squeeze(-1)  # (B, T)
        if key_padding_mask is not None:
            s = s.masked_fill(key_padding_mask, float("-inf"))
        w = torch.softmax(s, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (w * h).sum(dim=1)  # (B, D)


class AffectHead(nn.Module):
    """Pooling + MLP -> (valence, arousal) w [-1, 1]."""

    def __init__(self, d_model: int, d_hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.pool = AttentivePooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, 2),
        )

    def forward(self, enc: torch.Tensor, key_padding_mask: torch.Tensor | None = None):
        pooled = self.norm(self.pool(enc, key_padding_mask))
        return torch.tanh(self.mlp(pooled))  # (B, 2) = (V, A)


def ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Concordance Correlation Coefficient per batch (1D tensory)."""
    pm, tm = pred.mean(), target.mean()
    pv, tv = pred.var(unbiased=False), target.var(unbiased=False)
    cov = ((pred - pm) * (target - tm)).mean()
    return 2 * cov / (pv + tv + (pm - tm) ** 2 + 1e-8)


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - mean(CCC_v, CCC_a); pred/target (B, 2)."""
    return 1 - 0.5 * (ccc(pred[:, 0], target[:, 0]) + ccc(pred[:, 1], target[:, 1]))


def load_frozen_encoder(checkpoint_path: str | Path, device: str = "cpu"):
    """Encoder z checkpointu train.py: zamrozony, eval. Zwraca (encoder, enc_dim,
    audio_cfg_dict, tokenizer_meta) -- cechy liczysz LogMelExtractorem z audio_cfg."""
    from model import ConformerTransducer

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_cfg = state["model_cfg"]
    # vocab z KSZTALTU wag jointa (V+1, joint_dim) -- odporne na braki w metadanych
    vocab_with_blank = state["model"]["joint.out.weight"].shape[0]
    model = ConformerTransducer(vocab_size=vocab_with_blank - 1, n_mels=80, **model_cfg)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if [k for k in missing if k.startswith("encoder.")]:
        raise RuntimeError(f"Brak wag encodera w checkpoincie: {missing[:5]}")
    enc = model.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, model_cfg["enc_dim"], state.get("audio_cfg", {}), state.get("tokenizer", {})
