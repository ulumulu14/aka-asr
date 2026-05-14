"""
Petla treningowa dla ConformerTransducer.

Spina:
    conformer_transducer_torch.py  -- model + rnnt_loss
    asr_preprocessing.py           -- dataset + collate_fn + tokenizer

Zawiera:
    - AdamW + Noam LR schedule (warmup + 1/sqrt decay) -- standard ASR
    - AMP (fp16 z GradScalerem albo bf16 bez)
    - Gradient clipping + gradient accumulation
    - Eval: walidacyjny RNN-T loss + WER przez greedy decode
    - Checkpointing (best by WER + last)
"""



from __future__ import annotations
from pathlib import Path

import logging
logger = logging.getLogger(__name__)
import math
import json
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import tempfile
import re
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib import tenumerate

import numpy as np
import pandas as pd
import soundfile as sf

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchsummary import summary


from model import ConformerTransducer, rnnt_loss
from preprocessing import (
    ASRDataset, CharTokenizer, AudioConfig,
    make_collate_fn, subsampled_length, Tokenizer,
)




_DTYPES = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("debug.log"),
        logging.StreamHandler()
    ]
)

# Config

@dataclass
class TrainConfig:
    # Optimizer
    peak_lr: float = 5e-4
    weight_decay: float = 1e-2
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9

    # Scheduler
    warmup_steps: int = 10_000

    # Trening
    max_epochs: int = 100
    grad_clip: float = 5.0
    accum_grad: int = 1
    batch_size: int = 16
    num_workers: int = 4

    # AMP
    use_amp: bool = True
    amp_dtype: str = 'bf16' #'bf16' | 'fp16' | 'fp32'

    # Log and eval
    log_every: int = 50
    eval_every_epochs: int = 1

    # Save ckpt
    ckpt_dir: str = 'checkpoints'
    keep_last: int = 3

    # Dekodowanie podczas eval
    max_symbols_per_step: int = 5

    # RNG
    seed: int = 42



# WER
def _edit_distance(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    if n == 0: return m
    if m == 0: return n
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
    """Word Error Rate. Dla CER wystarczy podzielic znakami zamiast spacjami."""
    total_dist = total_words = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        total_dist += _edit_distance(rw, hw)
        total_words += len(rw)
    return total_dist / max(1, total_words)


# Scheduler -- Noam (warmup + 1/sqrt(step) decay), znormalizowany do peak_lr

def make_noam_scheduler(optimizer, warmup_steps: int) -> LambdaLR:
    """factor(step) = warmup^0.5 * min(step^-0.5, step * warmup^-1.5)
    factor(warmup) = 1, czyli lr osiaga peak_lr na koncu warmupu i potem maleje 1/sqrt(step).
    """
    warmup = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        s = max(1, step)
        return warmup ** 0.5 * min(s ** -0.5, s * warmup ** -1.5)

    return LambdaLR(optimizer, lr_lambda)


# Greedy decode na batchu (z poszanowaniem logit_lengths)

@torch.no_grad()
def greedy_decode_batch(
        model: ConformerTransducer,
        audio_features: torch.Tensor,
        logit_lengths: torch.Tensor,
        sos_id: int = 0,
        blank_id: int = 0,
        max_symbols_per_step: int = 5
    ) -> list[list[int]]:

    """Decoduje caly batch, ale dla kazdego sampla iteruje tylko po jego
    rzeczywistej liczbie ramek (encoder output), bez decodowania paddingu.

    Uwaga: predictor jest przeliczany od zera po kazdym nowym tokenie.
    Dla produkcji uzyj cache stanu (lub stateless predictor + N-gram).
    """

    
    model.eval()
    device = audio_features.device
    B = audio_features.shape[0]

    f = model.encoder(audio_features)               # (B, T', d_enc)
    enc_lens = [int(L) for L in logit_lengths]

    all_hyps: list[list[int]] = []
    for b in range(B):
        T_b = enc_lens[b]
        f_b = f[b:b + 1, :T_b]                      # (1, T_b, d)
        hyp = torch.tensor([[sos_id]], dtype=torch.long, device=device)
        out_tokens: list[int] = []

        for t in range(T_b):
            f_t = f_b[:, t:t + 1, :]
            for _ in range(max_symbols_per_step):
                g = model.predictor(hyp)
                logit = model.joint(f_t, g[:, -1:, :])   # (1,1,1,V+1)
                pred = int(logit.argmax(dim=-1).item())

                if pred == blank_id:
                    break
                
                out_tokens.append(pred)
                hyp = torch.cat(
                    [hyp, torch.tensor([[pred]], dtype=torch.long, device=device)],
                    dim=1,
                )
        all_hyps.append(out_tokens)

    return all_hyps

from torch.utils.data import Sampler
import random

class LengthBucketBatchSampler:
    def __init__(self, durations, batch_size, shuffle=True):
        self.bs = batch_size
        self.shuffle = shuffle
        idx_sorted = sorted(range(len(durations)), key=lambda i: durations[i])
        self.batches = [idx_sorted[i:i+batch_size]
                        for i in range(0, len(idx_sorted), batch_size)]
    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            random.shuffle(order)
        for i in order:
            yield self.batches[i]
    def __len__(self):
        return len(self.batches)


# Eval (loss + WER)

@torch.no_grad()
def evaluate(
        model: ConformerTransducer,
        loader: DataLoader,
        tokenizer: Tokenizer,
        device: torch.device,
        max_symbols_per_step: int = 5
    ) -> dict:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    refs: list[str] = []
    hyps: list[str] = []

    for batch in tqdm(loader):
        audio  = batch['audio_features'].to(device)
        labels = batch['label_tokens'].to(device)
        targets = batch['targets'].to(device)
        logit_lengths  = batch['logit_lengths'].to(device)
        target_lengths = batch['target_lengths'].to(device)

        logits = model(audio, labels)
        loss = rnnt_loss(logits.float(), targets, logit_lengths, target_lengths, blank=0)
        #loss = rnnt_loss(logits, targets, logit_lengths, target_lengths, blank=0)
        bs = audio.size(0)
        total_loss   += loss.item() * bs
        total_samples += bs

        # Greedy decode + WER
        hyp_ids = greedy_decode_batch(
            model, audio, logit_lengths,
            max_symbols_per_step=max_symbols_per_step,
        )

        for i in range(bs):
            ref_ids = targets[i, :int(target_lengths[i])].tolist()
            refs.append(tokenizer.decode(ref_ids))
            hyps.append(tokenizer.decode(hyp_ids[i]))

    return {
        'loss': total_loss / max(1, total_samples),
        'wer' : compute_wer(refs, hyps),
        'examples': list(zip(refs[:3], hyps[:3])),
    }


# Checkpoint save

def save_ckpt(path: Path, *, model, optimizer, scheduler, scaler,
              step: int, epoch: int, best_wer: float, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model'      : model.state_dict(),
        'optimizer'  : optimizer.state_dict(),
        'scheduler'  : scheduler.state_dict(),
        'scaler'     : scaler.state_dict() if scaler is not None else None,
        'step'       : step,
        'epoch'      : epoch,
        'best_wer'   : best_wer,
        'cfg'        : cfg.__dict__,
    }, path)


def load_ckpt(path: Path, *, model, optimizer=None, scheduler=None, scaler=None,
              map_location='cpu') -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state['model'])
    if optimizer is not None and 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None and 'scheduler' in state:
        scheduler.load_state_dict(state['scheduler'])
    if scaler is not None and state.get('scaler') is not None:
        scaler.load_state_dict(state['scaler'])
    return state


def _cleanup_old_ckpts(ckpt_dir: Path, keep: int) -> None:
    snaps = sorted(ckpt_dir.glob('epoch_*.pt'),
                   key=lambda p: int(p.stem.split('_')[1]))
    for old in snaps[:-keep]:
        old.unlink(missing_ok=True)



# Traning loop
def train(
        model: ConformerTransducer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: Tokenizer,
        cfg: TrainConfig,
        device: torch.device | None = None
    ) -> None:

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    torch.manual_seed(cfg.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg.seed)

    # --- Optimizer: weight decay tylko na "tlustych" parametrach ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if p.ndim < 2 or n.endswith('.bias') or 'norm' in n.lower() or 'embed' in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = AdamW(
        [{'params': decay,    'weight_decay': cfg.weight_decay},
         {'params': no_decay, 'weight_decay': 0.0}],
        lr=cfg.peak_lr, betas=cfg.betas, eps=cfg.eps,
    )
    scheduler = make_noam_scheduler(optimizer, cfg.warmup_steps)

    # --- AMP ---
    amp_dtype = _DTYPES[cfg.amp_dtype]
    use_amp = cfg.use_amp and device.type == 'cuda' and amp_dtype != torch.float32
    # GradScaler tylko dla fp16; bf16 ma zakres wykladnika fp32, wiec niepotrzebny
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and amp_dtype == torch.float16))

    ckpt_dir = Path(cfg.ckpt_dir)
    best_wer = float('inf')
    global_step = 0
    t0 = time.time()

    logger.info('Trening...')
    try:
        for epoch in range(cfg.max_epochs):
            t0_epoch = time.time()
            model.train()
            running_loss = 0.0
            running_n = 0
            optimizer.zero_grad(set_to_none=True)

            #for batch_idx, batch in enumerate(tqdm(train_loader)):
            for batch_idx, batch in (enumerate(pbar := tqdm(train_loader))):
                audio  = batch['audio_features'].to(device, non_blocking=True)
                labels = batch['label_tokens'].to(device, non_blocking=True)
                targets   = batch['targets'].to(device, non_blocking=True)
                logit_lengths  = batch['logit_lengths'].to(device, non_blocking=True)
                target_lengths = batch['target_lengths'].to(device, non_blocking=True)

                # Forward w AMP, loss zawsze w fp32
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(audio, labels)
                loss = rnnt_loss(
                    logits.float(), targets, logit_lengths, target_lengths, blank=0,
                )

                running_loss += loss.item() * audio.size(0)
                running_n    += audio.size(0)
                avg = running_loss / max(1, running_n)
                lr = scheduler.get_last_lr()[0]

                pbar.set_description(f"[epka {epoch:3d} | step {global_step:7d}] | loss={avg:.4f}")

                if not torch.isfinite(loss):
                    # Skip step ale nie zerujemy stanu -- czesto efekt pojedynczego skoku
                    logger.info(f"[step {global_step}] loss={loss.item()} -- pomijam batch")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss / cfg.accum_grad).backward()

                running_loss += loss.item() * audio.size(0)
                running_n    += audio.size(0)

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
                        elapsed = time.time() - t0
                        epoch_time = time.time() - t0_epoch
                        logger.info(
                            f"[ep {epoch:3d} | step {global_step:7d}] "
                            f"loss={avg:.4f}  lr={lr:.2e}  "
                            f"|grad|={float(grad_norm):.2f}  "
                            f"global time={elapsed/60:.1f}m"
                            f"epoch time={epoch_time}"       
                        )
    except Exception as e:
        logger.error(e)

        # --- Eval ---
        #logger.info("Starting validation...")
        logger.info("Walidacja...")
        try:
            if (epoch + 1) % cfg.eval_every_epochs == 0:
                metrics = evaluate(
                    model, val_loader, tokenizer, device,
                                    max_symbols_per_step=cfg.max_symbols_per_step)
                logger.info(f"[epka {epoch:3d}] val_loss={metrics['loss']:.4f}  ")
                logger.info(f"val_WER={metrics['wer']:.2%}")
                for ref, hyp in metrics['examples']:
                    logger.info(f"REF: {ref!r}")
                    logger.info(f"HYP: {hyp!r}")

                # Best checkpoint
                if metrics['wer'] < best_wer:
                    best_wer = metrics['wer']
                    save_ckpt(ckpt_dir / 'best.pt',
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, step=global_step, epoch=epoch,
                            best_wer=best_wer, cfg=cfg)
                    logger.info(f"    -> nowy zajebisty WER = {best_wer:.2%}, zapisuje zajebisty best.pt")

            # Last N checkpoints
            save_ckpt(ckpt_dir / f'epoch_{epoch:04d}.pt',
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    scaler=scaler, step=global_step, epoch=epoch,
                    best_wer=best_wer, cfg=cfg)
            _cleanup_old_ckpts(ckpt_dir, cfg.keep_last)
        except Exception as e:
            logger.error(e)

    logger.info(f"\nnajzajebistrzy WER = {best_wer:.2%}")


if __name__ == '__main__':
    #in = input(f"Dawaj oś kurwa")


    

    # 1) Mini-manifest z dwoma "utterance'ami" (sinusy zamiast mowy)
    # tmp = Path(tempfile.mkdtemp())
    # sr  = 16000
    # paths = []
    # for i, dur in enumerate([1.5, 2.3, 1.8, 2.0]):
    #     n = int(sr * dur)
    #     t = np.linspace(0, dur, n, endpoint=False)
    #     wav = (0.1 * np.sin(2 * np.pi * (220 + 50 * i) * t)).astype(np.float32)
    #     p = tmp / f"sample_{i}.wav"
    #     sf.write(p, wav, sr)
    #     paths.append(p)

    #transcripts = ["ala ma kota", "kot ma ale", "ola pije sok", "kawa jest gorzka"]
    #train_manifest = tmp / "train_mainfest.jsonl"
    #val_manifest   = tmp / "val_manifest.jsonl"
    t = time.time()
    #batch_size = 32
    batch_size = 16
    
    
    # google colab paths
    train_manifest = "/content/cv-corpus-25.0-2026-03-09-pl/train_manifest.jsonl"
    val_manifest   = "/content/cv-corpus-25.0-2026-03-09-pl/val_manifest_smol.jsonl"

    # local paths
    #train_manifest = "D:\\kod\\src\\asr_pytorch_local\\train_manifest_22k_plPL.jsonl"
    #val_manifest  =  "D:\\kod\\src\\asr_pytorch_local\\val_manifest_9K_plPL.json"
    root = Path("path/to/parent")
    pattern = re.compile(r"^experiment_\d+$")

    #logger.info(torch.cuda.is_available())
    logger.info(f"CUDA diwajses: {torch.cuda.device_count()}")
    #logger.info(torch.cuda.current_device())
    #logger.info(torch.cuda.device(0))
    logger.info(f"{torch.cuda.get_device_name(torch.cuda.device_count()-1)}")

    #with open('config.yaml', 'r') as file:
    #    config = yaml.safe_load(file)

    #count = sum(1 for p in root.iterdir() if p.is_dir() and pattern.match(p.name))
    #print(count)


    #with open(train_manifest, 'w') as f:
    #    for p, txt, dur in zip(paths, transcripts, [1.5, 2.3, 1.8, 2.0]):
    #        f.write(json.dumps({"audio_path": str(p), "duration": dur, "text": txt}) + "\n")
    #with open(val_manifest, 'w') as f:
    #    for p, txt, dur in zip(paths[:2], transcripts[:2], [1.5, 2.3]):
    #        f.write(json.dumps({"audio_path": str(p), "duration": dur, "text": txt}) + "\n")

    # 2) Tokenizer
    tokenizer = CharTokenizer(alphabet="abcdefghijklmnoprstuwxyzqvąćęłńóśźż ")

    # 3) Datasety + loadery
    train_ds = ASRDataset(train_manifest, tokenizer, augment=True, max_duration=10)
    val_ds   = ASRDataset(val_manifest,   tokenizer, augment=False, max_duration=10)
    #train_ds.precompute_features("train_manifest_22k_plPL.jsonl", "features_cache/")
    collate  = make_collate_fn(blank_id=0)

    
    cfg = TrainConfig(
        peak_lr=1e-3, warmup_steps=15000,
        max_epochs=100, batch_size=batch_size, num_workers=8, # big
        #max_epochs=50, batch_size=batch_size, num_workers=0, # smol
        use_amp=True, amp_dtype='bf16',
        log_every=1, eval_every_epochs=1,
        ckpt_dir=f'checkpoints/conformer_m/experiment_{t}',
        max_symbols_per_step=3,
        weight_decay=1e-6, grad_clip=1.0,
    )
    #train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    bucket_sampler = LengthBucketBatchSampler(
        train_ds.durations, batch_size=batch_size, shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=bucket_sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Smol
    # model = ConformerTransducer(
    #     vocab_size=tokenizer.vocab_size,
    #     n_mels=80,
    #     enc_layers=2, enc_dim=64, enc_heads=4, enc_kernel=15,
    #     pred_layers=1, pred_dim=64, pred_heads=4, pred_kernel=7,
    #     joint_dim=128, dropout=0.1,
    # )

    # Medium model
    # 1) pred_layers = 2/4
    #2) 
    model = ConformerTransducer(
       vocab_size=tokenizer.vocab_size,
       n_mels=80,
       enc_layers=16, enc_dim=256, enc_heads=4, enc_kernel=31, # encoeder
       pred_layers=1, pred_dim=320, pred_heads=4, pred_kernel=15, # decoder
       joint_dim=320, dropout=0.1,
    )

    # Big mf model
    # model = ConformerTransducer(
    #     vocab_size=tokenizer.vocab_size,
    #     n_mels=80,
    #     enc_layers=17, enc_dim=512, enc_heads=8, enc_kernel=31,
    #     pred_layers=2, pred_dim=640, pred_heads=8, pred_kernel=15,
    #     joint_dim=640, dropout=0.1,
    # )

    #print(f"model params: {sum(p.numel() for p in model.parameters()):,}")
    #summary(model, input_size=(2))
    #print(train_ds)

    # 5) Smoke test treningu (2 epoki, krotkie)
    # cfg = TrainConfig(
    #     peak_lr=3e-4, warmup_steps=20,
    #     max_epochs=2, batch_size=128, num_workers=0,
    #     use_amp=False, amp_dtype='fp32',
    #     log_every=1, eval_every_epochs=1,
    #     ckpt_dir=str(tmp / 'ckpts'),
    #     max_symbols_per_step=3,
    # )



    try:
        train(model, train_loader, val_loader, tokenizer, cfg)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Smoke test pominiety: {e}  (zainstaluj torchaudio do treningu)")
