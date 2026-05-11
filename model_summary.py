import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchinfo import summary

from model import ConformerTransducer  # plik z modelem


def main():
    # --- Hiperparametry (jak w domyślnej konfiguracji modelu) -------------

    # model = ConformerTransducer(
    #     vocab_size=tokenizer.vocab_size,
    #     n_mels=80,
    #     enc_layers=2, enc_dim=64, enc_heads=4, enc_kernel=15,
    #     pred_layers=1, pred_dim=64, pred_heads=4, pred_kernel=7,
    #     joint_dim=128, dropout=0.1,
    # )
    VOCAB_SIZE = 1024
    N_MELS     = 80

    # Big
    #ENC_LAYERS, ENC_DIM, ENC_HEADS, ENC_KERNEL = 16, 256, 4, 31
    #PRED_LAYERS, PRED_DIM, PRED_HEADS, PRED_KERNEL = 2, 256, 4, 15
    #JOINT_DIM = 320

    # Smol
    ENC_LAYERS, ENC_DIM, ENC_HEADS, ENC_KERNEL = 2, 64, 4, 15
    PRED_LAYERS, PRED_DIM, PRED_HEADS, PRED_KERNEL = 1, 64, 4, 7
    JOINT_DIM = 128


    # --- Kształty dummy inputów ------------------------------------------
    B = 32
    T = 400          # 4s audio przy 100 fps
    U = 30           # długość label-sequence (bez SOS)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Model -----------------------------------------------------------
    model = ConformerTransducer(
        vocab_size=VOCAB_SIZE,
        n_mels=N_MELS,
        enc_layers=ENC_LAYERS, enc_dim=ENC_DIM,
        enc_heads=ENC_HEADS, enc_kernel=ENC_KERNEL,
        pred_layers=PRED_LAYERS, pred_dim=PRED_DIM,
        pred_heads=PRED_HEADS, pred_kernel=PRED_KERNEL,
        joint_dim=JOINT_DIM,
        causal_encoder=False,
    ).to(device)

    # --- Dummy inputs (właściwe dtypes!) ---------------------------------
    audio_features = torch.randn(B, T, N_MELS, device=device)
    label_tokens   = torch.randint(0, VOCAB_SIZE + 1, (B, U + 1), device=device)
    # UWAGA: przy obecnej implementacji encodera maska (B, T) crashuje MHSA
    # po subsamplingu (oczekuje (B, T/4)). Daję None, żeby summary się policzył.
    summary(
        model,
        input_data=[audio_features, label_tokens],
        col_names=("input_size", "output_size", "num_params", "mult_adds"),
        depth=4,
        device=device,
        verbose=1,
    )

    # --- Quick param count (sanity check) --------------------------------
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal params:     {total/1e6:>7.2f} M")
    print(f"Trainable params: {trainable/1e6:>7.2f} M")


if __name__ == "__main__":
    main()