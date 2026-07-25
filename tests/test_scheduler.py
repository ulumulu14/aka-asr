"""Cosine LR scheduler: warmup -> peak -> cosinusowy zjazd do floora -> plasko.
W przeciwienstwie do Noama LR nie zdycha na dlugich runach (floor = min_lr_ratio)."""
import pytest
import torch

from train import make_cosine_scheduler


def _run(warmup, total, min_ratio, steps):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)  # base_lr=1 -> lr == mnoznik lambdy
    sched = make_cosine_scheduler(opt, warmup, total, min_ratio)
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        p.grad = torch.zeros_like(p)
        opt.step()
        sched.step()
    return lrs


def test_cosine_warmup_peak_floor():
    lrs = _run(warmup=10, total=100, min_ratio=0.1, steps=120)
    assert lrs[0] == pytest.approx(0.1, abs=1e-6)   # start warmupu (1/10)
    assert lrs[9] == pytest.approx(1.0, abs=1e-6)   # koniec warmupu = peak
    assert max(lrs) == pytest.approx(1.0, abs=1e-6)  # nie przekracza peak
    assert lrs[99] == pytest.approx(0.1, abs=0.02)  # ~floor przy total
    assert lrs[115] == pytest.approx(0.1, abs=1e-9)  # floor po total (plasko)


def test_cosine_monotonic_decay():
    lrs = _run(warmup=10, total=100, min_ratio=0.1, steps=100)
    decay = lrs[10:]  # po warmupie
    assert all(b <= a + 1e-12 for a, b in zip(decay, decay[1:]))
