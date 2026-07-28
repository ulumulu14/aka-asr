import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchaudio.functional import rnnt_loss as ta_rnnt_loss

# Rozmiary modelu wybierane przez --model_size / config
MODEL_SIZES = {
    "smol": dict(
        enc_layers=2,
        enc_dim=64,
        enc_heads=4,
        enc_kernel=15,
        pred_layers=1,
        pred_dim=64,
        pred_heads=4,
        pred_kernel=7,
        joint_dim=128,
    ),
    "mid": dict(
        enc_layers=16,
        enc_dim=256,
        enc_heads=4,
        enc_kernel=31,
        pred_layers=1,
        pred_dim=320,
        pred_heads=4,
        pred_kernel=15,
        joint_dim=320,
    ),
    "big": dict(
        enc_layers=17,
        enc_dim=512,
        enc_heads=8,
        enc_kernel=31,
        pred_layers=2,
        pred_dim=640,
        pred_heads=8,
        pred_kernel=15,
        joint_dim=640,
    ),
    # Preset wiekszy niz big (~319M): nastepny krok drabinki skalowania.
    # enc_dim musi dzielic sie przez enc_heads (tu 768/12=64 na glowe).
    "giga": dict(
        enc_layers=20,
        enc_dim=768,
        enc_heads=12,
        enc_kernel=31,
        pred_layers=2,
        pred_dim=768,
        pred_heads=8,
        pred_kernel=15,
        joint_dim=768,
    ),
    # Rozmiar do definiowania przez uzytkownika: baza = big, a wlasciwa
    # architektura przychodzi z configu -- `model.dims: {enc_dim: 1024,
    # enc_layers: 24, ...}` podmienia dowolne pola (patrz train.py; size=custom
    # bez model.dims to blad na starcie, zeby nie trenowac po cichu kopii biga).
    "custom": dict(
        enc_layers=17,
        enc_dim=512,
        enc_heads=8,
        enc_kernel=31,
        pred_layers=2,
        pred_dim=640,
        pred_heads=8,
        pred_kernel=15,
        joint_dim=640,
    ),
}


class FeedForwardModule(nn.Module):

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * expansion)
        self.act = nn.SiLU()  # Swish
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model * expansion, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class ConvolutionModule(nn.Module):

    def __init__(
        self, d_model: int, kernel_size: int = 31, dropout: float = 0.1, causal: bool = False
    ):
        super().__init__()
        assert kernel_size % 2 == 1 or causal, "kernel_size powinno byc nieparzyste dla non-causal"
        self.causal = causal
        self.kernel_size = kernel_size

        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        if causal:
            # padding = 0 + manualny lewostronny F.pad w forward
            self.dw = nn.Conv1d(
                d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=0
            )
            # BN psuje causality na inferencji single-frame -> LN
            self.norm = nn.LayerNorm(d_model)
        else:
            self.dw = nn.Conv1d(
                d_model,
                d_model,
                kernel_size=kernel_size,
                groups=d_model,
                padding=(kernel_size - 1) // 2,
            )
            self.norm = nn.BatchNorm1d(d_model)

        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, C, T)
        x = self.pw1(x)  # (B, 2C, T)
        x = F.glu(x, dim=1)  # (B, C, T)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0))  # lewostronny pad
        x = self.dw(x)  # (B, C, T)

        if isinstance(self.norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)

        x = self.act(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)  # (B, T, C)
        return self.drop(x)


class MultiHeadSelfAttentionModule(nn.Module):

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None, causal: bool = False
    ) -> torch.Tensor:
        x_ln = self.ln(x)
        attn_mask = None

        if causal:
            L = x_ln.size(1)
            attn_mask = torch.triu(
                torch.full((L, L), float("-inf"), device=x_ln.device, dtype=x_ln.dtype),
                diagonal=1,
            )

        out, _ = self.mha(
            x_ln,
            x_ln,
            x_ln,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        return self.drop(out)


class ConformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal
        self.ffn1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout, causal=causal)
        self.ffn2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask, causal=self.causal)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.ln_out(x)


class Conv2dSubsampling(nn.Module):

    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        # Po dwoch stride-2 conv2d: F' = ceil(ceil(F/2)/2)
        f_out = ((n_mels + 1) // 2 + 1) // 2
        self.proj = nn.Linear(d_model * f_out, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  # (B, C, T', F')
        b, c, t, f_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f_dim)
        return self.drop(self.proj(x))


class SinusoidalPositionalEncoding(nn.Module):
    """
    Absolutne sinusoidalne pozycje dodawane po subsamplingu -- nn.MultiheadAttention
    sama z siebie nie zna kolejnosci ramek.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._build(max_len), persistent=False)

    def _build(self, length: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(length, self.d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: self.d_model // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        if L > self.pe.size(0):
            self.pe = self._build(L).to(device=x.device)
        pe = self.pe[:L].to(device=x.device, dtype=x.dtype)
        return self.dropout(x * self.xscale + pe)


class ConformerAudioEncoder(nn.Module):

    def __init__(
        self,
        n_mels: int = 80,
        num_layers: int = 16,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        subsample: bool = True,
        causal: bool = False,
        use_pos_enc: bool = True,
    ):
        super().__init__()
        self.subsample = (
            Conv2dSubsampling(n_mels, d_model, dropout) if subsample else nn.Linear(n_mels, d_model)
        )
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout) if use_pos_enc else None
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    num_heads=num_heads,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                    causal=causal,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, *, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.subsample(x)
        if key_padding_mask is not None and key_padding_mask.shape[1] != x.shape[1]:
            raise ValueError(
                f"key_padding_mask ma dlugosc {key_padding_mask.shape[1]}, "
                f"a wyjscie subsamplingu {x.shape[1]} - maska musi byc policzona "
                f"po subsamplingu (subsampled_length)"
            )
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        for block in self.blocks:
            # Zerowanie paddingu, zeby conv i BatchNorm nie mieszaly smieci
            # z ramek dopelnienia do prawdziwych ramek
            if key_padding_mask is not None:
                x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            x = block(x, key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class ConformerLabelEncoder(nn.Module):

    def __init__(
        self,
        vocab_size_with_blank: int,
        num_layers: int = 2,
        d_model: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size_with_blank, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model,
                    num_heads=num_heads,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                    causal=True,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x)
        return x


class JointNetwork(nn.Module):

    def __init__(self, enc_dim: int, pred_dim: int, vocab_size: int, joint_dim: int = 320):
        super().__init__()
        self.proj_enc = nn.Linear(enc_dim, joint_dim)
        self.proj_pred = nn.Linear(pred_dim, joint_dim)
        self.out = nn.Linear(joint_dim, vocab_size + 1)  # +1 = blank

    def forward(self, f, g):
        fe = self.proj_enc(f).unsqueeze(2)  # (B, T, 1, J)
        gp = self.proj_pred(g).unsqueeze(1)  # (B, 1, U+1, J)
        return self.out(torch.tanh(fe + gp))  # (B, T, U+1, V+1)


class ConformerTransducer(nn.Module):

    def __init__(
        self,
        vocab_size: int,  # bez blanka
        n_mels: int = 80,
        enc_layers: int = 16,
        enc_dim: int = 256,
        enc_heads: int = 4,
        enc_kernel: int = 31,
        pred_layers: int = 2,
        pred_dim: int = 256,
        pred_heads: int = 4,
        pred_kernel: int = 15,
        joint_dim: int = 320,
        dropout: float = 0.1,
        causal_encoder: bool = False,
        use_pos_enc: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size

        self.encoder = ConformerAudioEncoder(
            n_mels=n_mels,
            num_layers=enc_layers,
            d_model=enc_dim,
            num_heads=enc_heads,
            conv_kernel=enc_kernel,
            dropout=dropout,
            causal=causal_encoder,
            use_pos_enc=use_pos_enc,
        )
        self.predictor = ConformerLabelEncoder(
            vocab_size_with_blank=vocab_size + 1,
            num_layers=pred_layers,
            d_model=pred_dim,
            num_heads=pred_heads,
            conv_kernel=pred_kernel,
            dropout=dropout,
        )
        self.joint = JointNetwork(
            enc_dim=enc_dim,
            pred_dim=pred_dim,
            vocab_size=vocab_size,
            joint_dim=joint_dim,
        )
        # Pomocnicza glowica CTC na encoderze (aux loss stabilizujacy trening RNN-T)
        self.ctc_head = nn.Linear(enc_dim, vocab_size + 1)

    def forward(
        self,
        audio_features: torch.Tensor,
        label_tokens: torch.Tensor,
        audio_padding_mask: torch.Tensor | None = None,
        return_encoder_out: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        audio_features : (B, T, n_mels)
        label_tokens   : (B, U+1)
        zwraca logits  : (B, T/4, U+1, V+1)
        return_encoder_out=True zwraca dodatkowo wyjscie encodera (B, T/4, d_enc),
        np. do policzenia aux CTC bez drugiego przebiegu encodera.
        """
        f = self.encoder(audio_features, key_padding_mask=audio_padding_mask)
        g = self.predictor(label_tokens)
        logits = self.joint(f, g)
        if return_encoder_out:
            return logits, f
        return logits


def rnnt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:

    return ta_rnnt_loss(
        logits=logits,
        targets=targets.to(torch.int32),
        logit_lengths=logit_lengths.to(torch.int32),
        target_lengths=target_lengths.to(torch.int32),
        blank=blank,
        reduction=reduction,
    )
