#!/usr/bin/env python3
"""
Dekodowanie RNN-T ponad greedy: beam search z opcjonalna plytka fuzja LM.

Narzedzie ewaluacyjne/inferencyjne -- walidacja treningowa zostaje na szybkim
greedy z train.py. Beam z logsumexp-merge identycznych hipotez, limit emisji
na ramke jak w greedy (max_symbols_per_step).

    from decoding import beam_search_decode_batch
    hyps = beam_search_decode_batch(model, feats, logit_lengths, beam_size=4)

Plytka fuzja (shallow fusion): podaj obiekt z metoda
    logprobs(prefix: list[int]) -> torch.Tensor   # (V+1,)
w przestrzeni ID tokenizera ASR (0 = blank, nigdy nie odpytywany):
    score = log P_asr + lm_weight * log P_lm + length_bonus * |tokeny|
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class PrefixLM(Protocol):
    """Interfejs LM do plytkiej fuzji (n-gram / neural nad tokenami ASR)."""

    def logprobs(self, prefix: list[int]) -> torch.Tensor: ...  # (V+1,)


@dataclass
class _Hyp:
    tokens: tuple[int, ...]
    logp: float
    g: torch.Tensor  # (1, 1, pred_dim) -- wyjscie predictora dla prefiksu


def _merge(pool: dict[tuple[int, ...], _Hyp], cand: _Hyp) -> None:
    """Scala hipotezy o identycznych tokenach (logsumexp po sciezkach w latticu)."""
    ex = pool.get(cand.tokens)
    if ex is None:
        pool[cand.tokens] = cand
    else:
        m = max(ex.logp, cand.logp)
        ex.logp = m + math.log(math.exp(ex.logp - m) + math.exp(cand.logp - m))


def _predict_g(model, tokens: tuple[int, ...], device, sos_id: int) -> torch.Tensor:
    inp = torch.tensor([[sos_id, *tokens]], dtype=torch.long, device=device)
    return model.predictor(inp)[:, -1:, :]


@torch.no_grad()
def beam_search_decode_batch(
    model,
    audio_features: torch.Tensor,
    logit_lengths: torch.Tensor,
    audio_padding_mask: torch.Tensor | None = None,
    beam_size: int = 4,
    max_symbols_per_step: int = 3,
    sos_id: int = 0,
    blank_id: int = 0,
    lm: PrefixLM | None = None,
    lm_weight: float = 0.0,
    length_bonus: float = 0.0,
) -> list[list[int]]:
    model.eval()
    device = audio_features.device
    B = audio_features.shape[0]

    if audio_padding_mask is None:
        T_sub_max = int(logit_lengths.max())
        audio_padding_mask = (
            torch.arange(T_sub_max, device=device)[None, :]
            >= logit_lengths.to(device)[:, None]
        )

    f = model.encoder(audio_features, key_padding_mask=audio_padding_mask)  # (B, T', d)

    results: list[list[int]] = []
    for b in range(B):
        T_b = int(logit_lengths[b])
        f_b = f[b : b + 1, :T_b]
        beam = [_Hyp((), 0.0, _predict_g(model, (), device, sos_id))]

        for t in range(T_b):
            f_t = f_b[:, t : t + 1, :]  # (1, 1, d)
            ended: dict[tuple[int, ...], _Hyp] = {}  # skonsumowaly ramke (blank)
            active = beam

            for _ in range(max_symbols_per_step):
                if not active:
                    break
                g = torch.cat([h.g for h in active], dim=0)  # (K, 1, d)
                logits = model.joint(f_t.expand(len(active), -1, -1), g)  # (K,1,1,V+1)
                lp = F.log_softmax(logits.reshape(len(active), -1).float(), dim=-1)

                cands: dict[tuple[int, ...], _Hyp] = {}
                for i, h in enumerate(active):
                    _merge(ended, _Hyp(h.tokens, h.logp + float(lp[i, blank_id]), h.g))
                    lm_lp = None
                    if lm is not None and lm_weight != 0.0:
                        lm_lp = lm.logprobs(list(h.tokens))
                    top = torch.topk(lp[i], k=min(beam_size + 1, lp.shape[1]))
                    for score, tok in zip(top.values.tolist(), top.indices.tolist()):
                        if tok == blank_id:
                            continue
                        s = h.logp + score + length_bonus
                        if lm_lp is not None:
                            s += lm_weight * float(lm_lp[tok])
                        # g rodzica to placeholder -- przeliczany nizej dla przezywajacych
                        _merge(cands, _Hyp(h.tokens + (tok,), s, h.g))

                active = sorted(cands.values(), key=lambda h: h.logp, reverse=True)[:beam_size]
                for h in active:
                    h.g = _predict_g(model, h.tokens, device, sos_id)
            # hipotezy wciaz aktywne po limicie emisji: ich sciezka blank juz jest
            # w `ended` (dokladana w kazdej rundzie), wiec po prostu je porzucamy

            beam = sorted(ended.values(), key=lambda h: h.logp, reverse=True)[:beam_size]

        best = max(beam, key=lambda h: h.logp)
        results.append(list(best.tokens))

    return results
