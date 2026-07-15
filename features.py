"""
Jednorazowa ekstrakcja cech log-mel do plikow .npy.

Uzycie:
    python features.py train_manifest.jsonl features_cache/ --workers 8

Potem w configu ustaw data.feature_cache_dir na ten katalog -- DataLoader
bedzie ladowal gotowe cechy zamiast liczyc mel-spectrogramy w locie.
"""

import click

from preprocessing import ASRDataset
from train import setup_logging


@click.command()
@click.argument("manifest_path", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
@click.option("--workers", default=4, help="Liczba procesow ekstrakcji")
def main(manifest_path: str, out_dir: str, workers: int):
    setup_logging()
    ASRDataset.precompute_features(manifest_path, out_dir, n_workers=workers)


if __name__ == "__main__":
    main()
