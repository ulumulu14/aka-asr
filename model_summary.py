import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import click
import torch
from torchinfo import summary

from model import MODEL_SIZES, ConformerTransducer


@click.command()
@click.option("--size", type=click.Choice(list(MODEL_SIZES)), default="mid")
@click.option("--vocab-size", default=37, help="Rozmiar slownika bez blanka (char PL = 37)")
def main(size: str, vocab_size: int):
    N_MELS = 80
    B = 8
    T = 400  # 4s audio przy 100 fps
    U = 30   # dlugosc label-sequence (bez SOS)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ConformerTransducer(
        vocab_size=vocab_size,
        n_mels=N_MELS,
        **MODEL_SIZES[size],
    ).to(device)

    audio_features = torch.randn(B, T, N_MELS, device=device)
    label_tokens = torch.randint(0, vocab_size + 1, (B, U + 1), device=device)
    summary(
        model,
        input_data=[audio_features, label_tokens],
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        device=device,
        verbose=1,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total / 1e6:>7.2f} M")
    print(f"Trainable params: {trainable / 1e6:>7.2f} M")


if __name__ == "__main__":
    main()
