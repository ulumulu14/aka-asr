"""
Preprocessing danych ASR -> format wejscia treningowego dla ConformerTransducer.

Pipeline:
    wav -> log-mel (librosa) -> CMVN -> SpecAugment -> tensor (T, n_mels)
    text -> tokenizer -> ID (z offsetem +1, bo 0 = blank)
    batch -> padding + dlugosci -> dict gotowy dla forward() i rnnt_loss()

Format manifestu (JSONL, jeden sample per linia):
    {"audio_path": "/path/to/file.wav", "duration": 3.45, "text": "transkrypcja"}

Uwaga praktyczna:
    librosa jest wygodna ale wolna (CPU, numpy). W produkcji rozwaz:
      - pre-ekstrakcja cech do .npy raz a dobrze (idiom NeMo/ESPnet)
      - torchaudio.transforms.MelSpectrogram na GPU (10-50x szybciej online)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms


 
# Konfiguracja front-endu akustycznego
 

@dataclass
class AudioConfig:
    sample_rate: int   = 16000
    n_fft: int         = 400      # 25 ms @ 16 kHz
    win_length: int    = 400
    hop_length: int    = 160      # 10 ms  -> 100 ramek/s
    n_mels: int        = 80
    fmin: float        = 0.0
    fmax: float | None = None     # None -> sample_rate / 2
    preemphasis: float = 0.97     # 0.0 wylacza
    log_eps: float     = 1e-10


# Ekstrakcja cech

def extract_logmel(wav_path: str | Path, cfg: AudioConfig = AudioConfig()) -> np.ndarray:
    #print(wav_path)
    if type(wav_path) is str:
        wav_path = Path(wav_path)

    # torchaudio extraction on GPU
    try:
        waveform, sample_rate = torchaudio.load(wav_path, normalize=True)
    except Exception as e:
        print(f'problem z otworzeniem pliku {str(wav_path)}')
        raise e
        
    mel_transform = transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        n_mels=cfg.n_mels,
        f_min=cfg.fmin,
        f_max=cfg.fmax if cfg.fmax is not None else cfg.sample_rate / 2,
        power=2.0,
        window_fn=torch.hann_window,
        center=True,
        pad_mode='reflect',
        norm='slaney',         # <-- KLUCZOWE
        mel_scale='slaney',
    )
    
    mel = mel_transform(waveform)  # (channel, n_mels, time)
    mel = mel.squeeze().cpu().numpy()

    log_mel = np.log(mel + cfg.log_eps)
    return log_mel.T.astype(np.float32)     # (T, n_mels)



def cmvn(features: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Per-utterance cepstral mean and variance normalization.

    Dla treningu z duzymi datasetami warto policzyc *globalne* statystyki
    (mean/std per kanal mel po calym treningu) i ich uzywac zamiast per-utt --
    wtedy CMVN jest deterministyczne miedzy treningiem a inferencja.
    """
    mean = features.mean(axis=0, keepdims=True)
    std  = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + eps)


# SpecAugment (Park et al., 2019)

class SpecAugment:
    """Frequency masking + time masking na log-mel.

    Domyslne parametry odpowiadaja policy "LD" z papera (LibriSpeech double).
    time_mask_ratio adaptuje max szerokosc maski czasowej do dlugosci utterance --
    zapobiega zamaskowaniu wiekszej czesci krotkich nagran.
    """
    def __init__(self,
                 freq_mask_param: int = 27,
                 time_mask_param: int = 100,
                 num_freq_masks: int = 2,
                 num_time_masks: int = 10,
                 time_mask_ratio: float | None = 0.05,
                 rng: np.random.Generator | None = None):
        self.F = freq_mask_param
        self.T = time_mask_param
        self.mF = num_freq_masks
        self.mT = num_time_masks
        self.p  = time_mask_ratio
        self.rng = rng if rng is not None else np.random.default_rng()

    def __call__(self, features: np.ndarray) -> np.ndarray:
        T, M = features.shape
        out = features.copy()

        # Freq masking
        for _ in range(self.mF):
            f = int(self.rng.integers(0, self.F + 1))
            if f == 0 or M - f <= 0:
                continue
            f0 = int(self.rng.integers(0, M - f))
            out[:, f0:f0 + f] = 0.0

        # Time masking z adaptacyjnym cap'em
        cap = self.T if self.p is None else min(self.T, int(self.p * T))
        for _ in range(self.mT):
            t = int(self.rng.integers(0, max(1, cap + 1)))
            if t == 0 or T - t <= 0:
                continue
            t0 = int(self.rng.integers(0, T - t))
            out[t0:t0 + t, :] = 0.0

        return out


 
# Tokenizer (interfejs + prosta implementacja znakowa)

class Tokenizer(Protocol):
    """Interfejs tokenizera. ID 0 jest zarezerwowane dla blanka -- tokeny zwracane
    przez encode() musza byc >= 1.
    """
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...      # bez blanka


class CharTokenizer:
    """Prosty tokenizer znakowy. Do produkcji uzyj SentencePiece BPE.

    Dla SentencePiece wystarczy lekki wrapper, ktory robi `[t + 1 for t in sp.encode(...)]`
    i przy decode odejmuje 1 (zeby zachowac konwencje blank=0).
    """
    def __init__(self, alphabet: str):
        # ID 0 zarezerwowane dla blanka, znaki dostaja ID 1..len(alphabet)
        self.itos = ['<blank>'] + list(alphabet)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.itos[i] for i in ids if i != 0)

    @property
    def vocab_size(self) -> int:
        # Zwracamy rozmiar BEZ blanka -- model dolicza +1 wewnetrznie
        return len(self.itos) - 1


 
# Helper: dlugosc po subsamplingu

def subsampled_length(t_in: int) -> int:
    """Dlugosc po Conv2dSubsampling (dwa stride-2 conv2d, kernel=3, padding=1)."""
    t1 = (t_in - 1) // 2 + 1
    t2 = (t1   - 1) // 2 + 1
    return t2


 
# Dataset
 
class ASRDataset(Dataset):
    """Czyta manifest JSONL, ekstraktuje cechy on-the-fly.

    Dla szybkiego treningu rozwaz wariant z pre-ekstrakcja cech do .npy --
    metoda `precompute_features` pokazuje jak.
    """
    def __init__(self,
                 manifest_path: str | Path,
                 tokenizer: Tokenizer,
                 audio_cfg: AudioConfig = AudioConfig(),
                 augment: bool = False,
                 max_duration: float | None = 20.0,
                 min_duration: float | None = 0.5):
        with open(manifest_path, encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        # Filtrowanie po dlugosci -- bardzo dlugie nagrania pcha pamiec joint network
        if max_duration is not None or min_duration is not None:
            entries = [
                e for e in entries
                if (min_duration is None or e.get('duration', 0) >= min_duration)
                and (max_duration is None or e.get('duration', 1e9) <= max_duration)
            ]

        # Sortowanie po duration -- przy bucketed samplerze mocno przyspiesza padding
        entries.sort(key=lambda e: e.get('duration', 0.0))

        self.entries = entries
        self.tokenizer = tokenizer
        self.cfg = audio_cfg
        self.spec_aug = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        feats = extract_logmel(entry['audio_path'], self.cfg)
        feats = cmvn(feats)
        if self.spec_aug is not None:
            feats = self.spec_aug(feats)

        tokens = self.tokenizer.encode(entry['text'].lower())
        if len(tokens) == 0:
            #raise ValueError(f"Pusta transkrypcja po tokenizacji: {entry}")
            print(f"Pusta transkrypcja po tokenizacji, chuj wie o co chodzi: {entry}")

        return {
            'features': torch.from_numpy(feats),                     # (T, n_mels)
            'tokens'  : torch.tensor(tokens, dtype=torch.long),      # (U,)
        }

    @staticmethod
    def precompute_features(manifest_path: str | Path,
                            out_dir: str | Path,
                            audio_cfg: AudioConfig = AudioConfig()) -> None:
        """Jednorazowa ekstrakcja cech do plikow .npy (znacznie szybszy DataLoader)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line)
                feats = extract_logmel(entry['audio_path'], audio_cfg)
                stem = Path(entry['audio_path']).stem
                np.save(out_dir / f"{stem}.npy", feats)


 
# Collate -- skleja batch w format wymagany przez model + rnnt_loss
 
def make_collate_fn(blank_id: int = 0):
    """Zwraca collate_fn produkujacy slownik:
        audio_features  : (B, T_max, n_mels)        float32
        label_tokens    : (B, U_max + 1)            long      -- dla predictora (z SOS)
        targets         : (B, U_max)                int32     -- dla rnnt_loss (bez SOS)
        logit_lengths   : (B,)                      int32     -- T po subsamplingu
        target_lengths  : (B,)                      int32
    """
    def collate(batch: list[dict]) -> dict:
        B = len(batch)
        feats_list  = [b['features'] for b in batch]
        tokens_list = [b['tokens']   for b in batch]

        feat_lens = torch.tensor([f.shape[0] for f in feats_list],  dtype=torch.int32)
        tok_lens  = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.int32)

        T_max  = int(feat_lens.max())
        U_max  = int(tok_lens.max())
        n_mels = feats_list[0].shape[1]

        # Padding cech (zerami -- po CMVN to "neutralna" wartosc; przy ASR padding
        # i tak powinien byc maskowany po stronie modelu przez audio_padding_mask)
        padded_feats = torch.zeros(B, T_max, n_mels, dtype=torch.float32)
        for i, f in enumerate(feats_list):
            padded_feats[i, :f.shape[0]] = f

        # Padding tokenow (zerami -- targets sa odczytywane tylko do target_lengths)
        padded_tokens = torch.zeros(B, U_max, dtype=torch.long)
        for i, t in enumerate(tokens_list):
            padded_tokens[i, :t.shape[0]] = t

        # Wejscie predictora: SOS (= blank_id) doklejony na poczatek
        sos = torch.full((B, 1), blank_id, dtype=torch.long)
        pred_input = torch.cat([sos, padded_tokens], dim=1)          # (B, U_max + 1)

        # Logit lengths -- T po dwoch stride-2 conv2d w Conv2dSubsampling
        logit_lens = torch.tensor(
            [subsampled_length(int(L)) for L in feat_lens],
            dtype=torch.int32,
        )

        return {
            'audio_features': padded_feats,
            'label_tokens'  : pred_input,
            'targets'       : padded_tokens.to(torch.int32),
            'logit_lengths' : logit_lens,
            'target_lengths': tok_lens,
        }

    return collate


 
# Demo end-to-end
 

#if __name__ == '__main__':
#    import tempfile
#
#    # 1) Symulujemy maly manifest + dwa wav-y (sinusy) zeby pokazac flow
#    tmp = Path(tempfile.mkdtemp())
##    sr = 16000
#
#    import soundfile as sf
#    paths = []
#    for i, dur in enumerate([1.5, 2.3]):
#        n = int(sr * dur)
#        t = np.linspace(0, dur, n, endpoint=False)
#        wav = 0.1 * np.sin(2 * np.pi * (220 + 110 * i) * t).astype(np.float32)
#        p = tmp / f"sample_{i}.wav"
#        sf.write(p, wav, sr)
#        paths.append(p)

    #manifest = "D:/kod/src/asr_pytorch/manifest.json"
    #with open(manifest, 'w') as f:
    #    f.write(json.dumps({"audio_path": str(paths[0]), "duration": 1.5, "text": "ala ma kota"}) + "\n")
    #    f.write(json.dumps({"audio_path": str(paths[1]), "duration": 2.3, "text": "kot ma ale"}) + "\n")

    # 2) Tokenizer + dataset + dataloader
 #   tokenizer = CharTokenizer(alphabet="abcdefghijklmnoprstuwxyzqv ")
 #   print(f"vocab_size (bez blanka): {tokenizer.vocab_size}")

 #   ds = ASRDataset(manifest, tokenizer, augment=True)
 #   print(f"|dataset| = {len(ds)}")

 #   sample = ds[0]
 #   print(f"sample features : {tuple(sample['features'].shape)}  dtype={sample['features'].dtype}")
 #   print(f"sample tokens   : {sample['tokens'].tolist()}")

#    # 3) Batchowanie
#    from torch.utils.data import DataLoader
#    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=make_collate_fn(blank_id=0))

#    batch = next(iter(loader))
#    for k, v in batch.items():
#        print(f"{k:>15s}: shape={tuple(v.shape)}  dtype={v.dtype}")

    # 4) Format zgodny z ConformerTransducer.forward + rnnt_loss
#    print("\nSygnatura wejsciowa modelu:")
#    print(f"  model(audio_features={tuple(batch['audio_features'].shape)}, "
#          f"label_tokens={tuple(batch['label_tokens'].shape)})")
#    print(f"  rnnt_loss(logits, targets={tuple(batch['targets'].shape)}, "
#          f"logit_lengths={batch['logit_lengths'].tolist()}, "
#          f"target_lengths={batch['target_lengths'].tolist()})")