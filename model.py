"""
Conformer-Transducer-Conformer (all-Conformer RNN-T) -- PyTorch.

Architektura (taka sama jak w wersji TF):
    AudioEncoder    : Conformer (non-causal lub causal dla streamingu)
    LabelEncoder    : Conformer causal (predictor)
    JointNetwork    : tanh(W_e * f_t + W_p * g_u) -> linear(V + 1)

Konwencja shape'ow:
    Wewnatrz blokow trzymamy (B, T, C). Conv1d wymaga (B, C, T) - transpose
    pojawia sie tylko wewnatrz ConvolutionModule i Conv2dSubsampling.

Wymagania:
    torch >= 1.10
    torchaudio >= 0.10    (rnnt_loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

# =========================================================================
# Bloki bazowe Conformera
# =========================================================================

class FeedForwardModule(nn.Module):
    """Macaron-style FFN: LN -> Linear(expand) -> Swish -> Drop -> Linear -> Drop."""
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln    = nn.LayerNorm(d_model)
        self.fc1   = nn.Linear(d_model, d_model * expansion)
        self.act   = nn.SiLU()                              # Swish
        self.drop1 = nn.Dropout(dropout)
        self.fc2   = nn.Linear(d_model * expansion, d_model)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln(x)
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x


class ConvolutionModule(nn.Module):
    """Modul konwolucji Conformera (operuje wewnetrznie w (B, C, T)):
        LN -> PW Conv (2C) -> GLU -> DW Conv -> Norm -> Swish -> PW Conv -> Drop.

    causal=True: lewostronny padding w DW conv + LayerNorm zamiast BatchNorm.
    """
    def __init__(self, d_model: int, kernel_size: int = 31,
                 dropout: float = 0.1, causal: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1 or causal, "kernel_size powinno byc nieparzyste dla non-causal"
        self.causal      = causal
        self.kernel_size = kernel_size

        self.ln  = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        if causal:
            # padding = 0 + manualny lewostronny F.pad w forward
            self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                groups=d_model, padding=0)
            # BN psuje causality na inferencji single-frame -> LN
            self.norm = nn.LayerNorm(d_model)
        else:
            self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                                groups=d_model, padding=(kernel_size - 1) // 2)
            self.norm = nn.BatchNorm1d(d_model)

        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x = self.ln(x)
        x = x.transpose(1, 2)                       # (B, C, T)
        x = self.pw1(x)                             # (B, 2C, T)
        x = F.glu(x, dim=1)                         # (B, C, T)
        if self.causal:
            x = F.pad(x, (self.kernel_size - 1, 0)) # lewostronny pad
        x = self.dw(x)                              # (B, C, T)

        if isinstance(self.norm, nn.LayerNorm):
            x = x.transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2)
        else:
            x = self.norm(x)

        x = self.act(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)                       # (B, T, C)
        return self.drop(x)


class MultiHeadSelfAttentionModule(nn.Module):
    """LN -> MHSA -> Dropout."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln  = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, *,
                key_padding_mask: torch.Tensor | None = None,
                causal: bool = False) -> torch.Tensor:
        x_ln = self.ln(x)

        # nn.MultiheadAttention: is_causal to tylko HINT -- maska musi byc
        # dostarczona jawnie. PyTorch >= 2.0 rzuca RuntimeError przy
        # is_causal=True i attn_mask=None.
        attn_mask = None
        if causal:
            L = x_ln.size(1)
            attn_mask = torch.triu(
                torch.full((L, L), float('-inf'),
                           device=x_ln.device, dtype=x_ln.dtype),
                diagonal=1,
            )

        out, _ = self.mha(
            x_ln, x_ln, x_ln,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        return self.drop(out)


class ConformerBlock(nn.Module):
    """Conformer block (Macaron):
        x = x + 0.5 * FFN1(x)
        x = x + MHSA(x)
        x = x + Conv(x)
        x = x + 0.5 * FFN2(x)
        x = LN(x)
    """
    def __init__(self, d_model: int, num_heads: int = 4, ff_expansion: int = 4,
                 conv_kernel: int = 31, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.ffn1   = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa   = MultiHeadSelfAttentionModule(d_model, num_heads, dropout)
        self.conv   = ConvolutionModule(d_model, conv_kernel, dropout, causal=causal)
        self.ffn2   = FeedForwardModule(d_model, ff_expansion, dropout)
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, *,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)
        x = x + self.mhsa(x, key_padding_mask=key_padding_mask, causal=self.causal)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.ln_out(x)


# =========================================================================
# Subsampling akustyczny (4x time stride)
# =========================================================================

class Conv2dSubsampling(nn.Module):
    """Standardowe 4x subsamplingu czasu (dwa stride-2 conv2d).

    (B, T, n_mels) -> (B, T/4, d_model).
    """
    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1,        d_model, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model,  d_model, kernel_size=3, stride=2, padding=1)
        # Po dwoch stride-2 conv2d: F' = ceil(ceil(F/2)/2)
        f_out = ((n_mels + 1) // 2 + 1) // 2
        self.proj = nn.Linear(d_model * f_out, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))                   # (B, C, T', F')
        b, c, t, f_dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(b, t, c * f_dim)
        return self.drop(self.proj(x))


# =========================================================================
# Encoder akustyczny i Predictor
# =========================================================================

class ConformerAudioEncoder(nn.Module):
    """Stos blokow Conformera nad cechami akustycznymi (np. log-mel)."""
    def __init__(self, n_mels: int = 80, num_layers: int = 16, d_model: int = 256,
                 num_heads: int = 4, conv_kernel: int = 31, dropout: float = 0.1,
                 subsample: bool = True, causal: bool = False):
        super().__init__()
        self.subsample = Conv2dSubsampling(n_mels, d_model, dropout) if subsample \
            else nn.Linear(n_mels, d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads=num_heads,
                           conv_kernel=conv_kernel, dropout=dropout, causal=causal)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, *,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.subsample(x)
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        return x


class ConformerLabelEncoder(nn.Module):
    """Predictor (label encoder) - causal Conformer nad ciagiem tokenow.

    Wejscie: (B, U+1) tokenow (z SOS na poczatku).
    Wyjscie: (B, U+1, d_model).
    """
    def __init__(self, vocab_size_with_blank: int, num_layers: int = 2,
                 d_model: int = 256, num_heads: int = 4, conv_kernel: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        self.embed      = nn.Embedding(vocab_size_with_blank, d_model)
        self.embed_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, num_heads=num_heads,
                           conv_kernel=conv_kernel, dropout=dropout, causal=True)
            for _ in range(num_layers)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed_drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x)
        return x


# =========================================================================
# Joint network (RNN-T)
# =========================================================================

class JointNetwork(nn.Module):
    """Klasyczny RNN-T joint:
        z_{t,u} = W_o * tanh(W_e * f_t + W_p * g_u)
        P(k | t,u) = softmax(z_{t,u}),  k in {0..V}  (0 = blank)

    f: (B, T, d_enc), g: (B, U, d_pred)  ->  logits: (B, T, U, V+1).
    """
    def __init__(self, enc_dim: int, pred_dim: int, vocab_size: int, joint_dim: int = 320):
        super().__init__()
        self.proj_enc  = nn.Linear(enc_dim,  joint_dim)
        self.proj_pred = nn.Linear(pred_dim, joint_dim)
        self.out       = nn.Linear(joint_dim, vocab_size + 1)   # +1 = blank

    def forward(self, f, g, chunk: int = 16):
        # f: (B, T, d_enc), g: (B, U+1, d_pred)
        fe = self.proj_enc(f)                       # (B, T, J)
        gp = self.proj_pred(g).unsqueeze(1)         # (B, 1, U+1, J)
        B, T, J = fe.shape
        outs = []
        for t0 in range(0, T, chunk):
            f_c = fe[:, t0:t0+chunk].unsqueeze(2)   # (B, c, 1, J)
            outs.append(self.out(torch.tanh(f_c + gp)))
        return torch.cat(outs, dim=1)


# =========================================================================
# Pelny model
# =========================================================================

class ConformerTransducer(nn.Module):
    """Conformer-Transducer-Conformer.

    Konwencja: blank_id = 0, SOS wspoldzieli embedding z blankiem.
    """
    def __init__(self,
                 vocab_size: int,                    # bez blanka
                 n_mels: int = 80,
                 enc_layers: int = 16, enc_dim: int = 256, enc_heads: int = 4, enc_kernel: int = 31,
                 pred_layers: int = 2, pred_dim: int = 256, pred_heads: int = 4, pred_kernel: int = 15,
                 joint_dim: int = 320, dropout: float = 0.1, causal_encoder: bool = False):
        super().__init__()
        self.vocab_size = vocab_size

        self.encoder = ConformerAudioEncoder(
            n_mels=n_mels, num_layers=enc_layers, d_model=enc_dim,
            num_heads=enc_heads, conv_kernel=enc_kernel,
            dropout=dropout, causal=causal_encoder,
        )
        self.predictor = ConformerLabelEncoder(
            vocab_size_with_blank=vocab_size + 1,
            num_layers=pred_layers, d_model=pred_dim,
            num_heads=pred_heads, conv_kernel=pred_kernel, dropout=dropout,
        )
        self.joint = JointNetwork(
            enc_dim=enc_dim, pred_dim=pred_dim,
            vocab_size=vocab_size, joint_dim=joint_dim,
        )

    def forward(self, audio_features: torch.Tensor, label_tokens: torch.Tensor,
                audio_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        audio_features : (B, T, n_mels)
        label_tokens   : (B, U+1)            -- z SOS (np. blank) na poczatku
        zwraca logits  : (B, T/4, U+1, V+1)
        """
        #print(audio_features.shape)
        #print(label_tokens.shape)
        #print(audio_padding_mask.shape)
        f = self.encoder(audio_features, key_padding_mask=audio_padding_mask)
        g = self.predictor(label_tokens)
        return self.joint(f, g)


# =========================================================================
# Loss RNN-T (torchaudio)
# =========================================================================

def rnnt_loss(logits: torch.Tensor,
              targets: torch.Tensor,
              logit_lengths: torch.Tensor,
              target_lengths: torch.Tensor,
              blank: int = 0,
              reduction: str = 'mean') -> torch.Tensor:
    """Wrapper na torchaudio.functional.rnnt_loss.

    Argumenty:
        logits         : (B, T, U+1, V+1)   float32
        targets        : (B, U)             int32, bez SOS, bez blanka
        logit_lengths  : (B,)               int32, T po subsamplingu
        target_lengths : (B,)               int32, U
    """
    from torchaudio.functional import rnnt_loss as ta_rnnt_loss
    return ta_rnnt_loss(
        logits=logits,
        targets=targets.to(torch.int32),
        logit_lengths=logit_lengths.to(torch.int32),
        target_lengths=target_lengths.to(torch.int32),
        blank=blank,
        reduction=reduction,
    )


# =========================================================================
# Greedy decoding (referencyjny; do produkcji uzyj beam search + cache stanu)
# =========================================================================

@torch.no_grad()
def greedy_decode(model: ConformerTransducer,
                  audio_features: torch.Tensor,
                  sos_id: int = 0,
                  blank_id: int = 0,
                  max_symbols_per_step: int = 5) -> list[int]:
    """Greedy frame-synchronous RNN-T decoding -- pojedynczy przyklad (B=1)."""
    model.eval()
    device = audio_features.device

    f = model.encoder(audio_features)                       # (1, T', d)
    T = f.shape[1]

    hyp = torch.tensor([[sos_id]], dtype=torch.long, device=device)
    out_tokens: list[int] = []

    for t in range(T):
        f_t = f[:, t:t + 1, :]                              # (1, 1, d)
        for _ in range(max_symbols_per_step):
            g = model.predictor(hyp)                        # (1, U, d)
            logit = model.joint(f_t, g[:, -1:, :])          # (1, 1, 1, V+1)
            pred = int(logit.argmax(dim=-1).item())
            if pred == blank_id:
                break
            out_tokens.append(pred)
            hyp = torch.cat(
                [hyp, torch.tensor([[pred]], dtype=torch.long, device=device)],
                dim=1,
            )
    return out_tokens


# Sanity check

if __name__ == '__main__':
    torch.manual_seed(0)

    B, T_in, n_mels = 2, 200, 80
    U, V = 30, 1024

    model = ConformerTransducer(
        vocab_size=V, n_mels=n_mels,
        enc_layers=4,  enc_dim=144, enc_heads=4,  enc_kernel=31,
        pred_layers=2, pred_dim=144, pred_heads=4, pred_kernel=15,
        joint_dim=320, dropout=0.1,
    )

    audio  = torch.randn(B, T_in, n_mels)
    labels = torch.randint(0, V, (B, U))
    sos = torch.zeros(B, 1, dtype=torch.long)
    pred_in = torch.cat([sos, labels], dim=1)               

    logits = model(audio, pred_in)
    print(f"logits shape : {tuple(logits.shape)}")
    print(f"params       : {sum(p.numel() for p in model.parameters()):,}")
    

    # Loss
    try:
        T_out = logits.shape[1]
        logit_lengths  = torch.full((B,), T_out, dtype=torch.int32)
        target_lengths = torch.full((B,), U,     dtype=torch.int32)
        loss = rnnt_loss(logits, labels, logit_lengths, target_lengths, blank=0)
        print(f"rnnt_loss    : {loss.item():.4f}")
    except ImportError:
        print("torchaudio niedostepny -- pomijam test loss.")