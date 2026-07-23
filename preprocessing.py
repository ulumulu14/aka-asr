"""
Preprocessing

Pipeline:
    wav/flac/mp3 -> resample 16k + mono -> log-mel -> CMVN -> SpecAugment -> tensor (T, n_mels)
    text -> normalizacja -> tokenizer -> ID
    batch -> padding + dlugosci

Format manifestu (JSONL, jeden sample per linia):
    {"audio_path": "/path/to/file.wav", "duration": 3.45, "text": "transkrypcja"}
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio import transforms
from tqdm import tqdm

try:
    import soundfile as sf
except ImportError:
    sf = None

logger = logging.getLogger("asr")

# Polski alfabet + litery obce (q, v, x) spotykane w nazwach wlasnych
POLISH_ALPHABET = "aąbcćdeęfghijklłmnńoópqrsśtuvwxyzźż "

_DIGIT_RE = re.compile(r"\d")
_DASH_RE = re.compile(r"[-–—]")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400  # 25 ms @ 16 kHz
    win_length: int = 400
    hop_length: int = 160  # 10 ms  -> 100 ramek/s
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float | None = None  # None -> sample_rate / 2
    preemphasis: float = 0.97  # 0.0 wylacza
    log_eps: float = 1e-10


# Ekstrakcja cech


class LogMelExtractor:
    """
    Log-mel z pliku audio. Jedna instancja MelSpectrogram (filterbank liczony raz),
    downmix do mono, resampling do cfg.sample_rate i opcjonalna preemfaza.
    """

    def __init__(self, cfg: AudioConfig = AudioConfig()):
        self.cfg = cfg
        self.mel = transforms.MelSpectrogram(
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
            pad_mode="reflect",
            norm="slaney",
            mel_scale="slaney",
        )
        self._resamplers: dict[int, transforms.Resample] = {}

    @staticmethod
    def _load_audio(wav_path: str | Path) -> tuple[torch.Tensor, int]:
        """
        Zwraca (kanaly, T) float32 w [-1, 1]. Preferuje soundfile -- od
        torchaudio 2.9 torchaudio.load wymaga torchcodec + systemowego ffmpeg.
        """
        if sf is not None:
            data, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=True)
            return torch.from_numpy(data.T), sample_rate
        return torchaudio.load(str(wav_path), normalize=True)

    def load_waveform(self, wav_path: str | Path) -> torch.Tensor:
        """Mono (1, T) float32 w [-1, 1], zresamplowane do cfg.sample_rate."""
        try:
            waveform, sample_rate = self._load_audio(wav_path)
        except Exception:
            logger.error(f"Nie mozna otworzyc pliku audio: {wav_path}")
            raise

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.cfg.sample_rate:
            resampler = self._resamplers.get(sample_rate)
            if resampler is None:
                resampler = transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.cfg.sample_rate
                )
                self._resamplers[sample_rate] = resampler
            waveform = resampler(waveform)
        return waveform

    def features_from_waveform(self, waveform: torch.Tensor) -> np.ndarray:
        """Waveform mono (1, T) @ cfg.sample_rate -> log-mel (T, n_mels)."""
        # Pusty/uszkodzony plik albo bledna duracja w manifescie: MelSpectrogram
        # z center=True/reflect wymaga > n_fft//2 probek. Jasny blad zamiast
        # kryptycznego 'cannot reshape tensor of 0 elements'.
        if waveform.numel() < self.cfg.n_fft:
            raise ValueError(
                f"Waveform za krotki: {waveform.numel()} probek < n_fft={self.cfg.n_fft}"
            )
        if self.cfg.preemphasis:
            waveform = torch.cat(
                [
                    waveform[:, :1],
                    waveform[:, 1:] - self.cfg.preemphasis * waveform[:, :-1],
                ],
                dim=1,
            )

        mel = self.mel(waveform)  # (1, n_mels, T)
        log_mel = torch.log(mel.squeeze(0) + self.cfg.log_eps)
        return log_mel.T.numpy().astype(np.float32)  # (T, n_mels)

    def __call__(self, wav_path: str | Path) -> np.ndarray:
        return self.features_from_waveform(self.load_waveform(wav_path))


def extract_logmel(wav_path: str | Path, cfg: AudioConfig = AudioConfig()) -> np.ndarray:
    """Wrapper dla pojedynczych plikow; w petli uzywaj LogMelExtractor."""
    return LogMelExtractor(cfg)(wav_path)


def cmvn(features: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Per-utterance cepstral mean and variance normalization.
    """
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return (features - mean) / (std + eps)


# SpecAugment (Park et al., 2019)


def _lazy_worker_rng(rng: np.random.Generator | None) -> np.random.Generator:
    """
    Generator seedowany per proces z globalnego np.random (DataLoader seeduje
    go osobno w kazdym workerze). Eager default_rng() w __init__ bylby
    spicklowany do wszystkich workerow z identycznym stanem -> te same maski.
    """
    if rng is None:
        rng = np.random.default_rng(np.random.randint(0, 2**31 - 1))
    return rng


class SpecAugment:
    """
    Frequency masking + time masking na log-mel.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 10,
        time_mask_ratio: float | None = 0.05,
        rng: np.random.Generator | None = None,
    ):
        self.F = freq_mask_param
        self.T = time_mask_param
        self.mF = num_freq_masks
        self.mT = num_time_masks
        self.p = time_mask_ratio
        self.rng = rng  # None -> leniwie per worker w __call__

    def __call__(self, features: np.ndarray) -> np.ndarray:
        self.rng = _lazy_worker_rng(self.rng)
        T, M = features.shape
        out = features.copy()

        # Freq masking
        for _ in range(self.mF):
            f = int(self.rng.integers(0, self.F + 1))
            if f == 0 or M - f <= 0:
                continue
            f0 = int(self.rng.integers(0, M - f))
            out[:, f0 : f0 + f] = 0.0

        # Time masking z adaptacyjnym cap'em
        cap = self.T if self.p is None else min(self.T, int(self.p * T))
        for _ in range(self.mT):
            t = int(self.rng.integers(0, max(1, cap + 1)))
            if t == 0 or T - t <= 0:
                continue
            t0 = int(self.rng.integers(0, T - t))
            out[t0 : t0 + t, :] = 0.0

        return out


# Augmentacja waveformu: szum + pogłos (RIR)


class WaveformAugment:
    """
    Szum addytywny przy losowym SNR + pogłos (splot z losowym RIR) na waveformie,
    przed ekstrakcja cech. Dane np. z OpenSLR 28 (RIRS_NOISES): pointsource_noises
    jako noise_dir, simulated_rirs jako rir_dir.

    Augmentowany sample wymaga ekstrakcji log-mel on-the-fly (omija cache cech),
    wiec ASRDataset stosuje ja tylko dla ulamka sampli (waveform_augment_prob).
    """

    AUDIO_EXTS = (".wav", ".flac", ".ogg", ".opus")

    def __init__(
        self,
        noise_dir: str | Path | None = None,
        rir_dir: str | Path | None = None,
        sample_rate: int = 16000,
        noise_prob: float = 0.9,
        rir_prob: float = 0.5,
        snr_db: tuple[float, float] = (5.0, 25.0),
    ):
        if sf is None:
            raise ImportError("WaveformAugment wymaga soundfile: pip install soundfile")
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.rir_prob = rir_prob
        self.snr_db = snr_db
        self.noise_files = self._scan(noise_dir)
        self.rir_files = self._scan(rir_dir)
        if not self.noise_files and not self.rir_files:
            raise ValueError("WaveformAugment: brak plikow szumu i RIR (podaj noise_dir/rir_dir)")
        self.rng: np.random.Generator | None = None  # leniwie per worker

    @classmethod
    def _scan(cls, root: str | Path | None) -> list[Path]:
        if root is None:
            return []
        root = Path(root).expanduser()
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in cls.AUDIO_EXTS)
        if not files:
            raise ValueError(f"WaveformAugment: brak plikow audio w {root}")
        return files

    def _load(self, path: Path) -> np.ndarray:
        """Mono float32 @ self.sample_rate."""
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = data.mean(axis=1)
        if sr != self.sample_rate:
            wav_t = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, self.sample_rate
            )
            wav = wav_t.squeeze(0).numpy()
        return wav

    def _apply_rir(self, wav: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        rir = self._load(self.rir_files[int(rng.integers(len(self.rir_files)))])
        rir_t = torch.from_numpy(rir).unsqueeze(0)
        # Wyrownanie do direct path (pik RIR-a), zeby nie przesuwac audio wzgledem
        # transkrypcji; wyjscie przyciete do dlugosci wejscia
        peak = int(np.argmax(np.abs(rir)))
        out = torchaudio.functional.fftconvolve(wav, rir_t, mode="full")
        out = out[:, peak : peak + wav.shape[1]]
        # RIR-y maja rozne wzmocnienia -> przywroc energie oryginalu
        rms_in = wav.pow(2).mean().sqrt()
        rms_out = out.pow(2).mean().sqrt()
        if rms_out > 1e-8:
            out = out * (rms_in / rms_out)
        return out

    def _mix_noise(self, wav: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        noise = self._load(self.noise_files[int(rng.integers(len(self.noise_files)))])
        T = wav.shape[1]
        if len(noise) < T:
            noise = np.tile(noise, T // len(noise) + 1)
        start = int(rng.integers(0, len(noise) - T + 1))
        noise_t = torch.from_numpy(noise[start : start + T]).unsqueeze(0)

        p_sig = wav.pow(2).mean()
        p_noise = noise_t.pow(2).mean()
        if p_noise < 1e-10 or p_sig < 1e-10:
            return wav
        snr = float(rng.uniform(*self.snr_db))
        scale = torch.sqrt(p_sig / (p_noise * 10 ** (snr / 10)))
        return wav + scale * noise_t

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform mono (1, T) -> zaugmentowany waveform (1, T)."""
        self.rng = _lazy_worker_rng(self.rng)
        rng = self.rng

        applied = False
        if self.rir_files and rng.random() < self.rir_prob:
            wav = self._apply_rir(wav, rng)
            applied = True
        if self.noise_files and rng.random() < self.noise_prob:
            wav = self._mix_noise(wav, rng)
            applied = True
        if not applied:
            # koszt ekstrakcji on-the-fly juz poniesiony -- niech cos wejdzie
            if self.noise_files:
                wav = self._mix_noise(wav, rng)
            else:
                wav = self._apply_rir(wav, rng)

        peak = wav.abs().max()
        if peak > 1.0:  # ochrona przed clippingiem po zmiksowaniu
            wav = wav / peak
        return wav


def speed_perturb_waveform(wav: torch.Tensor, sample_rate: int, factor: float) -> torch.Tensor:
    """
    Perturbacja tempa w stylu Kaldi (sox 'speed'): resampling z przeetykietowanym
    sample rate. Zmienia tempo I wysokosc; dlugosc wyjscia = T / factor.
    """
    if factor == 1.0:
        return wav
    return torchaudio.functional.resample(
        wav, orig_freq=int(round(sample_rate * factor)), new_freq=sample_rate
    )


# Normalizacja tekstu


def normalize_text(text: str) -> str:
    """lowercase, myslniki -> spacja, zbicie bialych znakow."""
    text = text.lower()
    text = _DASH_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


# Tokenizer (interfejs + implementacje)
class Tokenizer(Protocol):

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...  # bez blanka


class CharTokenizer:

    def __init__(self, alphabet: str = POLISH_ALPHABET):
        # ID 0 zarezerwowane dla blanka, znaki dostaja ID 1..len(alphabet)
        self.itos = ["<blank>"] + list(alphabet)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != 0)

    @property
    def vocab_size(self) -> int:
        # Zwracamy rozmiar BEZ blanka - model dolicza +1 wewnetrznie
        return len(self.itos) - 1


class SentencePieceTokenizer:
    """
    Tokenizer BPE (sentencepiece). ID 0 zarezerwowane dla blanka,
    ID sentencepiece przesuniete o +1. Model trenuje utils/train_bpe.py.
    """

    def __init__(self, model_path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError(
                "Tokenizer BPE wymaga sentencepiece: pip install sentencepiece"
            ) from e
        self.sp = spm.SentencePieceProcessor(model_file=str(Path(model_path).expanduser()))

    def encode(self, text: str) -> list[int]:
        return [i + 1 for i in self.sp.encode(text)]

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode([i - 1 for i in ids if i > 0])

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()


# Helper: dlugosc po subsamplingu
def subsampled_length(t_in: int) -> int:
    """
    Dlugosc po Conv2dSubsampling (dwa stride-2 conv2d, kernel=3, padding=1).
    """
    t1 = (t_in - 1) // 2 + 1
    t2 = (t1 - 1) // 2 + 1
    return t2


def feature_cache_name(audio_path: str | Path) -> str:
    """
    Nazwa pliku cache cech. Hash pelnej sciezki zapobiega kolizjom stemow
    miedzy datasetami (np. CV i MLS w jednym cache_dir).
    """
    resolved = str(Path(audio_path).expanduser().resolve())
    digest = hashlib.md5(resolved.encode("utf-8")).hexdigest()[:10]
    return f"{Path(audio_path).stem}_{digest}.npy"


# Precompute cech w osobnych procesach (module-level dla picklingu)
_PRECOMPUTE_EXTRACTOR: LogMelExtractor | None = None


def _precompute_init(cfg: AudioConfig) -> None:
    global _PRECOMPUTE_EXTRACTOR
    torch.set_num_threads(1)
    _PRECOMPUTE_EXTRACTOR = LogMelExtractor(cfg)


def _precompute_one(task: tuple[str, str]) -> None:
    audio_path, out_path = task
    np.save(out_path, _PRECOMPUTE_EXTRACTOR(audio_path))


# Dataset
class ASRDataset(Dataset):
    """
    Czyta manifest JSONL. Cechy laduje z cache .npy (cache_dir, patrz
    precompute_features), a gdy pliku nie ma -- ekstraktuje on-the-fly.

    Przy inicjalizacji odfiltrowuje sample:
    - poza zakresem [min_duration, max_duration]
    - z cyframi w transkrypcji (tokenizer po cichu by je zgubil)
    - z pusta transkrypcja po normalizacji i tokenizacji
    """

    # Ile kolejnych sampli sprobowac, gdy trafi sie wadliwy plik audio
    _GETITEM_MAX_RETRIES = 25

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: Tokenizer,
        audio_cfg: AudioConfig = AudioConfig(),
        augment: bool = False,
        max_duration: float | None = 20.0,
        min_duration: float | None = 0.5,
        cache_dir: str | Path | None = None,
        waveform_augment: "WaveformAugment | None" = None,
        waveform_augment_prob: float = 0.4,
        speed_perturb_factors: list[float] | tuple[float, ...] | None = None,
    ):
        manifest_path = Path(manifest_path).expanduser()
        with open(manifest_path, encoding="utf-8") as f:
            raw_entries = [json.loads(line) for line in f if line.strip()]

        entries = []
        n_duration = n_digits = n_empty = 0
        for e in raw_entries:
            d = e.get("duration", 0.0)
            if (min_duration is not None and d < min_duration) or (
                max_duration is not None and d > max_duration
            ):
                n_duration += 1
                continue
            text = normalize_text(e["text"])
            if _DIGIT_RE.search(text):
                n_digits += 1
                continue
            if len(tokenizer.encode(text)) == 0:
                n_empty += 1
                continue
            e["text"] = text
            entries.append(e)

        entries.sort(key=lambda e: e.get("duration", 0.0))
        logger.info(
            f"{manifest_path.name}: {len(entries)}/{len(raw_entries)} sampli "
            f"(odrzucone: {n_duration} dlugosc, {n_digits} cyfry w tekscie, "
            f"{n_empty} pusta transkrypcja)"
        )

        self.entries = entries
        self.tokenizer = tokenizer
        self.cfg = audio_cfg
        self.extractor = LogMelExtractor(audio_cfg)
        self.spec_aug = SpecAugment() if augment else None
        self.wave_aug = waveform_augment if augment else None
        self.wave_aug_prob = waveform_augment_prob
        # Perturbacja tempa: kazdy sample losuje czynnik z listy (1.0 = bez zmiany).
        # Duracje do bucketingu zostaja z manifestu (odchylka do ~11% -- w budzecie).
        self.speed_factors = (
            [float(f) for f in speed_perturb_factors]
            if (augment and speed_perturb_factors) else None
        )
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.durations = [e.get("duration", 0.0) for e in entries]

    def __len__(self) -> int:
        return len(self.entries)

    def _compute_features(self, entry: dict) -> np.ndarray:
        # Speed perturbation i augmentacja waveformu wymuszaja ekstrakcje
        # on-the-fly (cache trzyma czyste cechy) -- cache tylko bez obu
        factor = 1.0
        if self.speed_factors is not None:
            factor = float(np.random.choice(self.speed_factors))
        do_wave_aug = self.wave_aug is not None and np.random.random() < self.wave_aug_prob
        if factor != 1.0 or do_wave_aug:
            wav = self.extractor.load_waveform(entry["audio_path"])
            if factor != 1.0:
                wav = speed_perturb_waveform(wav, self.cfg.sample_rate, factor)
            if do_wave_aug:
                wav = self.wave_aug(wav)
            return self.extractor.features_from_waveform(wav)
        if self.cache_dir is not None:
            cache_path = self.cache_dir / feature_cache_name(entry["audio_path"])
            if cache_path.exists():
                return np.load(cache_path)
        return self.extractor(entry["audio_path"])

    def __getitem__(self, idx: int) -> dict:
        # Uszkodzone/puste pliki audio (bledna duracja w manifescie przechodzi
        # filtr min_duration) nie moga ubijac calego treningu -- pomijamy wadliwy
        # sample i bierzemy kolejny z tego samego bucketa dlugosci.
        n = len(self.entries)
        for offset in range(self._GETITEM_MAX_RETRIES):
            entry = self.entries[(idx + offset) % n]
            try:
                feats = self._compute_features(entry)
            except Exception as e:
                logger.warning(f"Pomijam wadliwy sample {entry['audio_path']}: {e}")
                continue

            feats = cmvn(feats)
            if self.spec_aug is not None:
                feats = self.spec_aug(feats)

            tokens = self.tokenizer.encode(entry["text"])  # tekst znormalizowany w __init__

            return {
                "features": torch.from_numpy(feats),  # (T, n_mels)
                "tokens": torch.tensor(tokens, dtype=torch.long),  # (U,)
            }

        raise RuntimeError(
            f"{self._GETITEM_MAX_RETRIES} kolejnych sampli od idx={idx} nie dalo sie "
            "wczytac -- prawdopodobnie systematyczny problem z danymi"
        )

    @staticmethod
    def precompute_features(
        manifest_path: str | Path,
        out_dir: str | Path,
        audio_cfg: AudioConfig = AudioConfig(),
        n_workers: int = 1,
    ) -> None:
        """Jednorazowa ekstrakcja cech do plikow .npy (znacznie szybszy DataLoader)."""
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(Path(manifest_path).expanduser(), encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]

        tasks = []
        for entry in entries:
            out_path = out_dir / feature_cache_name(entry["audio_path"])
            if not out_path.exists():
                tasks.append((entry["audio_path"], str(out_path)))
        logger.info(f"Do policzenia: {len(tasks)}/{len(entries)} plikow")

        if n_workers <= 1:
            _precompute_init(audio_cfg)
            for task in tqdm(tasks, desc="ekstrakcja cech", dynamic_ncols=True):
                _precompute_one(task)
        else:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(
                max_workers=n_workers, initializer=_precompute_init, initargs=(audio_cfg,)
            ) as pool:
                list(
                    tqdm(
                        pool.map(_precompute_one, tasks, chunksize=16),
                        total=len(tasks),
                        desc="ekstrakcja cech",
                        dynamic_ncols=True,
                    )
                )


# Collate - skleja batch w format wymagany przez model + rnnt_loss
class CollateFn:
    """
    Collate_fn produkujacy slownik:
    audio_features  : (B, T_max, n_mels)
    label_tokens    : (B, U_max + 1)
    targets         : (B, U_max)
    logit_lengths   : (B,)
    target_lengths  : (B,)

    Klasa (nie closure), bo workery DataLoadera musza ja spicklowac --
    od Pythona 3.14 domyslny start method na Linuksie to forkserver.
    """

    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        feats_list = [b["features"] for b in batch]
        tokens_list = [b["tokens"] for b in batch]

        feat_lens = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.int32)
        tok_lens = torch.tensor([t.shape[0] for t in tokens_list], dtype=torch.int32)

        T_max = int(feat_lens.max())
        U_max = int(tok_lens.max())
        n_mels = feats_list[0].shape[1]

        # Padding cech
        padded_feats = torch.zeros(B, T_max, n_mels, dtype=torch.float32)
        for i, f in enumerate(feats_list):
            padded_feats[i, : f.shape[0]] = f

        # Padding tokenow
        padded_tokens = torch.zeros(B, U_max, dtype=torch.long)
        for i, t in enumerate(tokens_list):
            padded_tokens[i, : t.shape[0]] = t

        # Wejscie predictora
        sos = torch.full((B, 1), self.blank_id, dtype=torch.long)
        pred_input = torch.cat([sos, padded_tokens], dim=1)  # (B, U_max + 1)

        # Logit lengths
        logit_lens = torch.tensor(
            [subsampled_length(int(L)) for L in feat_lens],
            dtype=torch.int32,
        )

        T_sub_max = int(logit_lens.max())
        audio_padding_mask = (
            torch.arange(T_sub_max)[None, :] >= logit_lens[:, None]
        )

        return {
            "audio_features": padded_feats,
            "label_tokens": pred_input,
            "targets": padded_tokens.to(torch.int32),
            "logit_lengths": logit_lens,
            "target_lengths": tok_lens,
            "audio_padding_mask": audio_padding_mask,
        }


def make_collate_fn(blank_id: int = 0) -> CollateFn:
    return CollateFn(blank_id)
