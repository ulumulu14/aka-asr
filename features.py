from preprocessing import ASRDataset


if __name__ == 'main':
    ds = ASRDataset(train_manifest, tokenizer, augment=True)
    ASRDataset.precompute_features("train_manifest_22k_plPL.jsonl", "features_cache/")