"""
Trening Conformer-Transducer (RNN-T).

Uzycie:
    python train.py configs/config_local.yaml --model_size smol --device cpu
    python train.py configs/config_runpod.yaml
    python train.py configs/config_runpod.yaml --resume checkpoints/conformer_mid/<exp>/last.pt
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MODEL_SIZES, ConformerTransducer, rnnt_loss
from preprocessing import (
    POLISH_ALPHABET,
    ASRDataset,
    AudioConfig,
    CharTokenizer,
    SentencePieceTokenizer,
    Tokenizer,
    WaveformAugment,
    make_collate_fn,
)

_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

logger = logging.getLogger("asr")


def setup_logging(log_file: Path | None = None) -> None:
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)


class CollapseAbort(Exception):
    """RNN-T utknal w blank-collapse -- przerwij trening (patrz collapse_abort_epochs)."""


@dataclass
class TrainConfig:
    # Optimizer
    peak_lr: float = 1e-3
    weight_decay: float = 1e-6
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9

    # Scheduler
    warmup_steps: int = 10_000

    # Trening
    max_epochs: int = 50
    grad_clip: float = 1.0
    accum_grad: int = 1
    batch_size: int = 16
    num_workers: int = 4
    max_batch_duration: float | None = None  # sekundy audio na batch (dynamiczny batch size)

    # Aux CTC na encoderze (0.0 wylacza)
    ctc_weight: float = 0.3
    # Label smoothing aux CTC: interpolacja z rozkladem jednostajnym (0.0 wylacza)
    ctc_label_smoothing: float = 0.0

    # AMP
    use_amp: bool = True
    amp_dtype: str = "bf16"  # 'bf16' | 'fp16' | 'fp32'

    # Log i eval
    log_every: int = 50
    eval_every_epochs: int = 1
    wer_max_samples: int | None = None  # ile sampli walidacji dekodowac do WER (None = wszystkie)
    tensorboard: bool = True

    # Checkpointy
    ckpt_dir: str = "checkpoints"
    keep_last: int = 3
    save_every_steps: int = 1000  # rolling last.pt w trakcie epoki (0 = wylaczone)

    # Dekodowanie podczas eval
    max_symbols_per_step: int = 3

    # Anty blank-collapse: przerwij trening, jesli po >= N epokach val_WER
    # wciaz ~100% (RNN-T utknal w "emituj tylko blank"; 0 = wylaczone).
    # Zdrowe runy big+BPE schodza ponizej 100% juz w epoce 0-1.
    collapse_abort_epochs: int = 0

    # Inne
    compile: bool = False
    seed: int = 42


class LengthBucketBatchSampler:
    """
    Batchuje sample o zblizonej dlugosci.

    Posortowane indeksy dzielone sa na okna po (batch_size * bucket_mult) sampli;
    przy shuffle=True sklad okien jest tasowany co epoke, wiec batche zmieniaja
    sklad miedzy epokami, a padding pozostaje maly.

    max_batch_duration (sekundy) wlacza dynamiczny batch size: do batcha wchodzi
    tyle sampli, by max_dur_w_batchu * liczba_sampli nie przekroczylo budzetu
    (przyblizenie kosztu po paddingu); batch_size dziala wtedy jako gorny limit.
    """

    def __init__(
        self,
        durations,
        batch_size,
        shuffle=True,
        max_batch_duration=None,
        bucket_mult=50,
        seed=42,
    ):
        self.durations = list(durations)
        self.bs = batch_size
        self.shuffle = shuffle
        self.max_batch_duration = max_batch_duration
        self.window = max(batch_size * bucket_mult, batch_size)
        self.seed = seed
        self.epoch = 0
        self.batches = self._make_batches(random.Random(seed))

    def _make_batches(self, rng: random.Random) -> list[list[int]]:
        idx = sorted(range(len(self.durations)), key=lambda i: self.durations[i])
        if self.shuffle:
            for w0 in range(0, len(idx), self.window):
                window = idx[w0 : w0 + self.window]
                rng.shuffle(window)
                idx[w0 : w0 + self.window] = window

        if self.max_batch_duration is None:
            return [idx[i : i + self.bs] for i in range(0, len(idx), self.bs)]

        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0.0
        for i in idx:
            dur = max(self.durations[i], 0.1)
            padded_cost = max(cur_max, dur) * (len(cur) + 1)
            if cur and (padded_cost > self.max_batch_duration or len(cur) >= self.bs):
                batches.append(cur)
                cur, cur_max = [], 0.0
            cur.append(i)
            cur_max = max(cur_max, dur)
        if cur:
            batches.append(cur)
        return batches

    def __iter__(self):
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            self.epoch += 1
            self.batches = self._make_batches(rng)
            rng.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# WER
def _edit_distance(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m]


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    total_dist = total_words = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        total_dist += _edit_distance(rw, hw)
        total_words += len(rw)
    return total_dist / max(1, total_words)


# Scheduler Noam (warmup + 1/sqrt(step) decay), znormalizowany do peak_lr
def make_noam_scheduler(optimizer, warmup_steps: int) -> LambdaLR:
    warmup = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        s = max(1, step)
        return warmup**0.5 * min(s**-0.5, s * warmup**-1.5)

    return LambdaLR(optimizer, lr_lambda)


# Greedy decode na batchu
@torch.no_grad()
def greedy_decode_batch(
    model: ConformerTransducer,
    audio_features: torch.Tensor,
    logit_lengths: torch.Tensor,
    audio_padding_mask: torch.Tensor | None = None,
    sos_id: int = 0,
    blank_id: int = 0,
    max_symbols_per_step: int = 3,
    return_scores: bool = False,
) -> list[list[int]] | tuple[list[list[int]], list[float]]:
    """
    return_scores=True zwraca tez sredni log-prob wyemitowanych tokenow na
    sample (proxy pewnosci, np. do filtrowania pseudo-labeli). Pusta hipoteza
    dostaje -inf.
    """
    model.eval()
    device = audio_features.device
    B = audio_features.shape[0]

    if audio_padding_mask is None:
        T_sub_max = int(logit_lengths.max())
        audio_padding_mask = (
            torch.arange(T_sub_max, device=device)[None, :] >= logit_lengths.to(device)[:, None]
        )

    f = model.encoder(audio_features, key_padding_mask=audio_padding_mask)  # (B, T', d_enc)

    all_hyps: list[list[int]] = []
    all_scores: list[float] = []
    for b in range(B):
        T_b = int(logit_lengths[b])
        f_b = f[b : b + 1, :T_b]  # (1, T_b, d)
        hyp = torch.tensor([[sos_id]], dtype=torch.long, device=device)
        # Predictor przeliczany tylko gdy hipoteza sie wydluza -- przy blanku
        # (zdecydowana wiekszosc ramek) g sie nie zmienia
        g_last = model.predictor(hyp)[:, -1:, :]
        out_tokens: list[int] = []
        tok_logps: list[float] = []

        for t in range(T_b):
            f_t = f_b[:, t : t + 1, :]
            for _ in range(max_symbols_per_step):
                logit = model.joint(f_t, g_last)  # (1,1,1,V+1)
                pred = int(logit.argmax(dim=-1))

                if pred == blank_id:
                    break

                if return_scores:
                    lp = F.log_softmax(logit.flatten().float(), dim=-1)
                    tok_logps.append(float(lp[pred]))
                out_tokens.append(pred)
                hyp = torch.cat(
                    [hyp, torch.tensor([[pred]], dtype=torch.long, device=device)],
                    dim=1,
                )
                g_last = model.predictor(hyp)[:, -1:, :]
        all_hyps.append(out_tokens)
        if return_scores:
            all_scores.append(sum(tok_logps) / len(tok_logps) if tok_logps else float("-inf"))

    if return_scores:
        return all_hyps, all_scores
    return all_hyps


# val (loss + WER)
@torch.no_grad()
def validate(
    model: ConformerTransducer,
    loader: DataLoader,
    tokenizer: Tokenizer,
    device: torch.device,
    max_symbols_per_step: int = 3,
    wer_max_samples: int | None = None,
) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    refs: list[str] = []
    hyps: list[str] = []

    for batch in tqdm(loader, desc="walidacja", dynamic_ncols=True, ascii=True):
        audio = batch["audio_features"].to(device)
        labels = batch["label_tokens"].to(device)
        targets = batch["targets"].to(device)
        logit_lengths = batch["logit_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        audio_padding_mask = batch["audio_padding_mask"].to(device)

        logits = model(audio, labels, audio_padding_mask=audio_padding_mask)
        loss = rnnt_loss(logits.float(), targets, logit_lengths, target_lengths, blank=0)
        bs = audio.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        # Greedy decode + WER (opcjonalnie tylko na czesci walidacji -- decode jest drogi)
        if wer_max_samples is None or len(refs) < wer_max_samples:
            hyp_ids = greedy_decode_batch(
                model,
                audio,
                logit_lengths,
                audio_padding_mask=audio_padding_mask,
                max_symbols_per_step=max_symbols_per_step,
            )
            for i in range(bs):
                ref_ids = targets[i, : int(target_lengths[i])].tolist()
                refs.append(tokenizer.decode(ref_ids))
                hyps.append(tokenizer.decode(hyp_ids[i]))

    return {
        "loss": total_loss / max(1, total_samples),
        "wer": compute_wer(refs, hyps) if refs else float("nan"),
        "examples": list(zip(refs[:3], hyps[:3])),
    }


# Checkpointy


def save_ckpt(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    best_wer: float,
    cfg: TrainConfig,
    meta: dict | None = None,
    epoch_finished: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = getattr(model, "_orig_mod", model)  # torch.compile wrapper
    state = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "epoch_finished": epoch_finished,
        "best_wer": best_wer,
        "cfg": cfg.__dict__,
    }
    if meta:
        state.update(meta)
    # Zapis atomiczny -- przerwanie w trakcie nie zostawia uszkodzonego pliku
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(state, tmp_path)
    tmp_path.replace(path)


def load_ckpt(
    path: Path, *, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"
) -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state


def _cleanup_old_ckpts(ckpt_dir: Path, keep: int) -> None:
    snaps = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    for old in snaps[:-keep]:
        old.unlink(missing_ok=True)


# Training loop
def train(
    model: ConformerTransducer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: Tokenizer,
    cfg: TrainConfig,
    device: torch.device | None = None,
    resume_state: dict | None = None,
    meta: dict | None = None,
) -> None:

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer: bez weight decay dla biasow, norm i embeddingow
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith(".bias") or "norm" in n.lower() or "embed" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.peak_lr,
        betas=cfg.betas,
        eps=cfg.eps,
    )
    scheduler = make_noam_scheduler(optimizer, cfg.warmup_steps)

    # --- AMP ---
    amp_dtype = _DTYPES[cfg.amp_dtype]
    use_amp = cfg.use_amp and device.type == "cuda" and amp_dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_wer = float("inf")
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        if resume_state.get("scaler") is not None:
            scaler.load_state_dict(resume_state["scaler"])
        global_step = resume_state.get("step", 0)
        best_wer = resume_state.get("best_wer", float("inf"))
        last_epoch = resume_state.get("epoch", -1)
        # Checkpoint z konca epoki -> nastepna; z polowy (last.pt) -> powtorz epoke
        start_epoch = last_epoch + 1 if resume_state.get("epoch_finished", True) else last_epoch
        logger.info(f"Wznowiono trening: epoka {start_epoch}, krok {global_step}")

    if cfg.compile:
        logger.info("torch.compile(model)...")
        model = torch.compile(model)

    writer = None
    if cfg.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(Path(cfg.ckpt_dir) / "tb"))
        except ImportError:
            logger.warning(
                "Pakiet tensorboard nie jest zainstalowany -- pomijam logowanie metryk "
                "(pip install tensorboard)"
            )

    ckpt_dir = Path(cfg.ckpt_dir)
    t0 = time.time()
    epoch = start_epoch
    n_oom_skipped = 0  # batche pominiete przez CUDA OOM (skrajnie dlugie/geste)

    logger.info("Start treningu")
    epoch_bar = tqdm(
        range(start_epoch, cfg.max_epochs),
        desc="epoki",
        unit="ep",
        position=0,
        leave=True,
        initial=start_epoch,
        total=cfg.max_epochs,
        dynamic_ncols=True,
        ascii=True,
    )
    try:
        for epoch in epoch_bar:
            model.train()
            running_loss = 0.0
            running_n = 0
            optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(train_loader, position=1, leave=False, dynamic_ncols=True, ascii=True)
            for batch_idx, batch in enumerate(pbar):
                try:
                    audio = batch["audio_features"].to(device, non_blocking=True)
                    labels = batch["label_tokens"].to(device, non_blocking=True)
                    targets = batch["targets"].to(device, non_blocking=True)
                    logit_lengths = batch["logit_lengths"].to(device, non_blocking=True)
                    target_lengths = batch["target_lengths"].to(device, non_blocking=True)
                    audio_padding_mask = batch["audio_padding_mask"].to(device, non_blocking=True)

                    # Forward w AMP, loss liczony w fp32
                    with torch.amp.autocast(
                        device_type=device.type, dtype=amp_dtype, enabled=use_amp
                    ):
                        logits, enc_out = model(
                            audio,
                            labels,
                            audio_padding_mask=audio_padding_mask,
                            return_encoder_out=True,
                        )
                    loss_rnnt = rnnt_loss(
                        logits.float(), targets, logit_lengths, target_lengths, blank=0
                    )
                    loss = loss_rnnt
                    loss_ctc = None
                    if cfg.ctc_weight > 0:
                        raw_model = getattr(model, "_orig_mod", model)
                        ctc_log_probs = F.log_softmax(
                            raw_model.ctc_head(enc_out.float()), dim=-1
                        ).transpose(
                            0, 1
                        )  # (T', B, V+1)
                        loss_ctc = F.ctc_loss(
                            ctc_log_probs,
                            targets,
                            logit_lengths,
                            target_lengths,
                            blank=0,
                            zero_infinity=True,
                        )
                        if cfg.ctc_label_smoothing > 0:
                            # interpolacja z entropia krzyzowa do rozkladu
                            # jednostajnego, maskowana po realnych dlugosciach T'
                            T_sub = ctc_log_probs.shape[0]
                            frame_mask = (
                                torch.arange(T_sub, device=ctc_log_probs.device)[:, None]
                                < logit_lengths[None, :]
                            )
                            ls_uniform = -(
                                ctc_log_probs.mean(dim=-1) * frame_mask
                            ).sum() / frame_mask.sum().clamp(min=1)
                            eps = cfg.ctc_label_smoothing
                            loss_ctc = (1 - eps) * loss_ctc + eps * ls_uniform
                        loss = loss + cfg.ctc_weight * loss_ctc

                    if not torch.isfinite(loss):
                        logger.warning(f"[krok {global_step}] loss={loss.item()} -- pomijam batch")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    running_loss += loss.item() * audio.size(0)
                    running_n += audio.size(0)
                    pbar.set_description(
                        f"[epoka {epoch:3d} | krok {global_step:7d}] "
                        f"loss={running_loss / max(1, running_n):.4f}"
                    )

                    scaler.scale(loss / cfg.accum_grad).backward()
                except RuntimeError as e:
                    # Skrajnie dlugi/gesty batch rozdyma tensor jointa (B,T',U,J) ponad
                    # VRAM. Zamiast ubijac cala sesje - pomijamy batch, czyscimy cache
                    # i lecimy dalej. Inne RuntimeError przepuszczamy.
                    if "out of memory" not in str(e).lower():
                        raise
                    n_oom_skipped += 1
                    logger.warning(
                        f"[krok {global_step}] CUDA OOM na batchu "
                        f"{tuple(batch['audio_features'].shape)} -- pomijam "
                        f"(lacznie pominietych: {n_oom_skipped})"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    audio = labels = targets = logit_lengths = target_lengths = None
                    audio_padding_mask = logits = enc_out = None
                    loss = loss_rnnt = loss_ctc = ctc_log_probs = None
                    torch.cuda.empty_cache()
                    continue

                if (batch_idx + 1) % cfg.accum_grad == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % cfg.log_every == 0:
                        lr = scheduler.get_last_lr()[0]
                        avg = running_loss / max(1, running_n)
                        logger.info(
                            f"[epoka {epoch:3d} | krok {global_step:7d}] "
                            f"loss={avg:.4f}  lr={lr:.2e}  "
                            f"|grad|={float(grad_norm):.2f}  "
                            f"czas={(time.time() - t0) / 60:.1f}m"
                        )
                        if writer is not None:
                            writer.add_scalar("train/loss", avg, global_step)
                            writer.add_scalar("train/loss_rnnt", loss_rnnt.item(), global_step)
                            if loss_ctc is not None:
                                writer.add_scalar("train/loss_ctc", loss_ctc.item(), global_step)
                            writer.add_scalar("train/lr", lr, global_step)
                            writer.add_scalar("train/grad_norm", float(grad_norm), global_step)

                    # Rolling checkpoint na wypadek padu maszyny (spot pody itp.)
                    if cfg.save_every_steps and global_step % cfg.save_every_steps == 0:
                        save_ckpt(
                            ckpt_dir / "last.pt",
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            step=global_step,
                            epoch=epoch,
                            best_wer=best_wer,
                            cfg=cfg,
                            meta=meta,
                            epoch_finished=False,
                        )

            if (epoch + 1) % cfg.eval_every_epochs == 0:
                logger.info("Walidacja...")
                metrics = validate(
                    model,
                    val_loader,
                    tokenizer,
                    device,
                    max_symbols_per_step=cfg.max_symbols_per_step,
                    wer_max_samples=cfg.wer_max_samples,
                )
                logger.info(
                    f"[epoka {epoch:3d}] val_loss={metrics['loss']:.4f}  "
                    f"val_WER={metrics['wer']:.2%}"
                )
                epoch_bar.set_postfix(
                    val_loss=f"{metrics['loss']:.3f}", val_wer=f"{metrics['wer']:.2%}"
                )
                for ref, hyp in metrics["examples"]:
                    logger.info(f"REF: {ref} | HYP: {hyp}")
                if writer is not None:
                    writer.add_scalar("val/loss", metrics["loss"], global_step)
                    writer.add_scalar("val/wer", metrics["wer"], global_step)

                # Best checkpoint
                if metrics["wer"] < best_wer:
                    best_wer = metrics["wer"]
                    save_ckpt(
                        ckpt_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        epoch=epoch,
                        best_wer=best_wer,
                        cfg=cfg,
                        meta=meta,
                    )
                    logger.info(f"Nowy najlepszy WER={best_wer:.2%} -- zapisano best.pt")

                # Anty blank-collapse: po N epokach wciaz ~100% WER = RNN-T utknal
                # w "emituj tylko blank" i nie wyjdzie. Zdrowy run schodzi ponizej
                # 100% w epoce 0-1 -- przerywamy, zeby nie palic godzin na loterii.
                if (
                    cfg.collapse_abort_epochs > 0
                    and epoch + 1 >= cfg.collapse_abort_epochs
                    and best_wer >= 0.99
                ):
                    raise CollapseAbort(
                        f"blank-collapse: best_WER={best_wer:.2%} po {epoch + 1} epokach "
                        f">= progu {cfg.collapse_abort_epochs} -- przerywam (odpal ponownie)"
                    )

                # Last N checkpoints
                save_ckpt(
                    ckpt_dir / f"epoch_{epoch:04d}.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=global_step,
                    epoch=epoch,
                    best_wer=best_wer,
                    cfg=cfg,
                    meta=meta,
                )
                _cleanup_old_ckpts(ckpt_dir, cfg.keep_last)

        logger.info(
            f"Koniec treningu, najlepszy WER={best_wer:.2%} "
            f"(batchy pominietych przez OOM: {n_oom_skipped})"
        )
    except (KeyboardInterrupt, CollapseAbort) as e:
        if isinstance(e, CollapseAbort):
            logger.warning(str(e))
        else:
            logger.warning("Przerwano recznie -- zapisuje last.pt")
        save_ckpt(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=global_step,
            epoch=epoch,
            best_wer=best_wer,
            cfg=cfg,
            meta=meta,
            epoch_finished=False,
        )
    except Exception:
        logger.exception("Trening przerwany bledem")
        raise
    finally:
        epoch_bar.close()
        if writer is not None:
            writer.close()


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as cf:
        return yaml.safe_load(cf)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--model-size",
    "--model_size",
    "model_size",
    type=click.Choice(list(MODEL_SIZES)),
    default=None,
    help="Rozmiar modelu (nadpisuje config)",
)
@click.option(
    "--device",
    "device_str",
    type=click.Choice(["auto", "cuda", "cpu"]),
    default="auto",
    help="Urzadzenie treningu (auto = cuda jesli dostepna)",
)
@click.option(
    "--resume",
    "resume_path",
    type=click.Path(exists=True),
    default=None,
    help="Checkpoint .pt do wznowienia treningu",
)
def main(config_path: str, model_size: str | None, device_str: str, resume_path: str | None):
    config = load_config(config_path)
    tr = config.get("training", {})
    data = config.get("data") or tr  # stare configi trzymaly sciezki w training:
    tok_cfg = config.get("tokenizer") or {}

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise click.ClickException("CUDA niedostepna -- uzyj --device cpu")

    model_size = model_size or config.get("model", {}).get("size", "mid")

    cfg = TrainConfig(
        peak_lr=float(tr.get("lr", 1e-3)),
        weight_decay=float(tr.get("weight_decay", 1e-6)),
        warmup_steps=int(tr.get("warmup_steps", 10_000)),
        max_epochs=int(tr.get("epochs", 50)),
        grad_clip=float(tr.get("grad_clip", 1.0)),
        accum_grad=int(tr.get("accum_grad", 1)),
        batch_size=int(tr.get("batch_size", 16)),
        num_workers=int(tr.get("n_workers", 4)),
        max_batch_duration=tr.get("max_batch_duration"),
        ctc_weight=float(tr.get("ctc_weight", 0.3)),
        ctc_label_smoothing=float(tr.get("ctc_label_smoothing", 0.0)),
        use_amp=bool(tr.get("use_amp", True)),
        amp_dtype=str(tr.get("amp_dtype", "bf16")),
        log_every=int(tr.get("log_every", 50)),
        eval_every_epochs=int(tr.get("eval_every_epochs", 1)),
        wer_max_samples=tr.get("wer_max_samples"),
        tensorboard=bool(tr.get("tensorboard", True)),
        keep_last=int(tr.get("keep_last", 3)),
        save_every_steps=int(tr.get("save_every_steps", 1000)),
        max_symbols_per_step=int(tr.get("max_symbols_per_step", 3)),
        collapse_abort_epochs=int(tr.get("collapse_abort_epochs", 0)),
        compile=bool(tr.get("compile", False)),
        seed=int(tr.get("seed", 42)),
    )

    exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_dir = (
        Path(tr.get("ckpt_root", "checkpoints")).expanduser() / f"conformer_{model_size}" / exp_name
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.ckpt_dir = str(ckpt_dir)
    setup_logging(ckpt_dir / "train.log")

    logger.info(f"Config: {config_path} | model: {model_size} | device: {device}")
    logger.info(f"Checkpointy: {ckpt_dir}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # Seed przed inicjalizacja wag
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)

    # Tokenizer
    if tok_cfg.get("type", "char") == "bpe":
        bpe_model_path = tok_cfg["bpe_model_path"]
        tokenizer: Tokenizer = SentencePieceTokenizer(bpe_model_path)
        tokenizer_meta = {"type": "bpe", "bpe_model_path": str(bpe_model_path)}
    else:
        alphabet = tok_cfg.get("alphabet", POLISH_ALPHABET)
        tokenizer = CharTokenizer(alphabet=alphabet)
        tokenizer_meta = {"type": "char", "alphabet": alphabet}
    logger.info(f"Tokenizer: {tokenizer_meta['type']}, vocab={tokenizer.vocab_size} (+blank)")

    # Datasety + loadery
    audio_cfg = AudioConfig()

    # Augmentacja audio (szum + RIR) - tylko train
    aug_cfg = config.get("augment") or {}
    wave_aug = None
    wave_aug_prob = float(aug_cfg.get("prob", 0.4))
    if aug_cfg.get("noise_dir") or aug_cfg.get("rir_dir"):
        wave_aug = WaveformAugment(
            noise_dir=aug_cfg.get("noise_dir"),
            rir_dir=aug_cfg.get("rir_dir"),
            sample_rate=audio_cfg.sample_rate,
            noise_prob=float(aug_cfg.get("noise_prob", 0.9)),
            rir_prob=float(aug_cfg.get("rir_prob", 0.5)),
            snr_db=tuple(aug_cfg.get("snr_db", (5.0, 25.0))),
        )
        logger.info(
            f"Augmentacja waveformu: {len(wave_aug.noise_files)} szumow, "
            f"{len(wave_aug.rir_files)} RIR-ow, prob={wave_aug_prob}, "
            f"SNR={wave_aug.snr_db} dB"
        )

    speed_factors = aug_cfg.get("speed_perturb")
    if speed_factors:
        logger.info(f"Speed perturbation: {speed_factors}")

    train_ds = ASRDataset(
        data["train_manifest_path"],
        tokenizer,
        audio_cfg=audio_cfg,
        augment=True,
        max_duration=data.get("max_audio_len", 20.0),
        min_duration=data.get("min_audio_len", 0.5),
        cache_dir=data.get("feature_cache_dir"),
        waveform_augment=wave_aug,
        waveform_augment_prob=wave_aug_prob,
        speed_perturb_factors=speed_factors,
    )
    val_ds = ASRDataset(
        data["val_manifest_path"],
        tokenizer,
        audio_cfg=audio_cfg,
        augment=False,
        max_duration=data.get("max_audio_len", 20.0),
        min_duration=data.get("min_audio_len", 0.5),
        cache_dir=data.get("feature_cache_dir"),
    )
    collate = make_collate_fn(blank_id=0)

    train_sampler = LengthBucketBatchSampler(
        train_ds.durations,
        cfg.batch_size,
        shuffle=True,
        max_batch_duration=cfg.max_batch_duration,
        seed=cfg.seed,
    )
    val_sampler = LengthBucketBatchSampler(
        val_ds.durations,
        cfg.batch_size,
        shuffle=False,
        max_batch_duration=cfg.max_batch_duration,
    )

    dl_kwargs: dict = dict(
        collate_fn=collate,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    if cfg.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **dl_kwargs)

    # Model (przy resume architektura z checkpointa)
    resume_state = None
    model_cfg = dict(MODEL_SIZES[model_size])
    if resume_path:
        resume_state = torch.load(resume_path, map_location="cpu")
        model_cfg = resume_state.get("model_cfg", model_cfg)
        logger.info(f"Wznawiam z {resume_path}")

    model = ConformerTransducer(
        vocab_size=tokenizer.vocab_size,
        n_mels=audio_cfg.n_mels,
        dropout=float(config.get("model", {}).get("dropout", 0.1)),
        **model_cfg,
    )
    logger.info(f"Parametry modelu: {sum(p.numel() for p in model.parameters()):,}")

    # Blank-bias init (tylko swiezy trening): ujemny bias blanku w jointcie zniecha
    # ca model do emitowania samego blanku na starcie. Wazne przy BPE -- krotkie U
    # daje wysoki stosunek blank:token, przez co "emituj blank" jest silnym
    # atraktorem (pusty greedy, val_WER~100% od ep0). 0.0 = bez zmiany.
    blank_bias = float(config.get("model", {}).get("blank_bias_init", 0.0))
    if resume_state is None and blank_bias != 0.0:
        with torch.no_grad():
            model.joint.out.bias[0] = blank_bias  # indeks 0 = blank
        logger.info(f"Blank-bias init: joint.out.bias[blank]={blank_bias}")

    # Metadane w checkpointcie - pozwalaja odtworzyc model bez tego configu
    meta = {
        "model_cfg": model_cfg,
        "tokenizer": tokenizer_meta,
        "audio_cfg": asdict(audio_cfg),
    }

    train(
        model,
        train_loader,
        val_loader,
        tokenizer,
        cfg,
        device=device,
        resume_state=resume_state,
        meta=meta,
    )


if __name__ == "__main__":
    main()
