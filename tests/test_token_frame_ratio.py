"""Filtr U > T': odrzuc sample gdzie tokeny (U) > ratio * ramki po subsamplingu (T').
Chroni RNN-T przed loss explosion na krotkich klipach z dlugim tekstem (blank-collapse)."""

import json

from preprocessing import ASRDataset, CharTokenizer, subsampled_length, AudioConfig


def _manifest(tmp_path, rows):
    m = tmp_path / "m.jsonl"
    m.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(m)


def test_rejects_text_longer_than_audio(tmp_path):
    cfg = AudioConfig()
    fps = cfg.sample_rate / cfg.hop_length
    # 2s audio -> T' = subsampled_length(200) ramek
    tsub = subsampled_length(int(2.0 * fps))
    tok = CharTokenizer()
    short_txt = "ala"                      # U=3, male
    long_txt = "a b c " * 40               # U ~ duzo tokenow >> T'
    rows = [
        {"audio_path": "/x/ok.wav", "duration": 2.0, "text": short_txt},
        {"audio_path": "/x/bad.wav", "duration": 2.0, "text": long_txt},
    ]
    assert len(tok.encode(long_txt)) > tsub  # sanity: dlugi tekst faktycznie U>T'
    ds = ASRDataset(_manifest(tmp_path, rows), tok, max_token_frame_ratio=1.0)
    paths = [e["audio_path"] for e in ds.entries]
    assert "/x/ok.wav" in paths and "/x/bad.wav" not in paths


def test_ratio_0_5_keeps_healthy(tmp_path):
    # zdrowa mowa U/T' ~0.2 -- prog 0.5 jej nie tyka
    tok = CharTokenizer()
    rows = [{"audio_path": f"/x/{i}.wav", "duration": 5.0, "text": "dzien dobry panstwu"} for i in range(3)]
    ds = ASRDataset(_manifest(tmp_path, rows), tok, max_token_frame_ratio=0.5)
    assert len(ds.entries) == 3


def test_high_ratio_disables_filter(tmp_path):
    tok = CharTokenizer()
    rows = [{"audio_path": "/x/dense.wav", "duration": 1.0, "text": "a b c " * 30}]
    off = ASRDataset(_manifest(tmp_path, rows), tok, max_token_frame_ratio=999.0)
    on = ASRDataset(_manifest(tmp_path, rows), tok, max_token_frame_ratio=1.0)
    assert len(off.entries) == 1 and len(on.entries) == 0
