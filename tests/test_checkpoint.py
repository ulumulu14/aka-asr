"""Round-trip checkpointu: metadane (model_cfg/tokenizer/audio_cfg) odtwarzaja
model i wczytuja wagi. Na tym kontrakcie stoi demo apka, harvest i evaluate_wer."""

from dataclasses import asdict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import MODEL_SIZES, ConformerTransducer
from preprocessing import POLISH_ALPHABET, AudioConfig, CharTokenizer
from train import TrainConfig, save_ckpt


def test_checkpoint_reconstruction(tmp_path):
    torch.manual_seed(0)
    tok = CharTokenizer()
    audio = AudioConfig()
    model_cfg = dict(MODEL_SIZES["smol"])
    model = ConformerTransducer(vocab_size=tok.vocab_size, n_mels=audio.n_mels, **model_cfg).eval()
    opt = AdamW(model.parameters(), lr=1e-3)
    sched = LambdaLR(opt, lambda _s: 1.0)
    meta = {
        "model_cfg": model_cfg,
        "tokenizer": {"type": "char", "alphabet": POLISH_ALPHABET},
        "audio_cfg": asdict(audio),
    }
    ckpt = tmp_path / "ckpt.pt"
    save_ckpt(
        ckpt,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=None,
        step=1,
        epoch=0,
        best_wer=1.0,
        cfg=TrainConfig(),
        meta=meta,
    )

    # odtworzenie DOKLADNIE jak transcriber/harvest/evaluate_wer
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    for key in ("model", "model_cfg", "tokenizer", "audio_cfg"):
        assert key in state
    rtok = CharTokenizer(alphabet=state["tokenizer"]["alphabet"])
    rcfg = AudioConfig(**state["audio_cfg"])
    rec = ConformerTransducer(
        vocab_size=rtok.vocab_size, n_mels=rcfg.n_mels, **state["model_cfg"]
    ).eval()
    rec.load_state_dict(state["model"])

    feats, labels = torch.randn(1, 32, 80), torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        assert torch.allclose(model(feats, labels), rec(feats, labels), atol=1e-5)
