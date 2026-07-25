"""Denylist uszkodzonego audio: wyklucza przy starcie, dopisuje nowe w getitem.
Osobny plik (nie mutacja manifestu). Bez bad_list_path -> zachowanie niezmienione."""

import json

from preprocessing import ASRDataset, CharTokenizer


def _manifest(tmp_path, paths):
    m = tmp_path / "m.jsonl"
    m.write_text(
        "\n".join(json.dumps({"audio_path": p, "duration": 2.0, "text": "ala ma kota"}) for p in paths),
        encoding="utf-8",
    )
    return str(m)


def test_denylist_excludes_at_start(tmp_path):
    paths = [f"/x/{i}.wav" for i in range(5)]
    deny = tmp_path / "bad.txt"
    deny.write_text("/x/1.wav\n/x/3.wav\n", encoding="utf-8")
    ds = ASRDataset(_manifest(tmp_path, paths), CharTokenizer(), bad_list_path=deny)
    got = sorted(e["audio_path"] for e in ds.entries)
    assert got == ["/x/0.wav", "/x/2.wav", "/x/4.wav"]  # 1 i 3 wykluczone


def test_no_bad_list_unchanged(tmp_path):
    paths = [f"/x/{i}.wav" for i in range(5)]
    ds = ASRDataset(_manifest(tmp_path, paths), CharTokenizer())  # bez bad_list_path
    assert len(ds.entries) == 5
    assert ds.bad_list_path is None


def test_mark_bad_appends_and_dedups(tmp_path):
    deny = tmp_path / "bad.txt"
    ds = ASRDataset(_manifest(tmp_path, ["/x/0.wav"]), CharTokenizer(), bad_list_path=deny)
    ds._mark_bad("/y/broken.wav")
    ds._mark_bad("/y/broken.wav")  # duplikat z tego procesu -> nie dopisuje 2x
    ds._mark_bad("/y/other.wav")
    lines = deny.read_text(encoding="utf-8").splitlines()
    assert lines == ["/y/broken.wav", "/y/other.wav"]


def test_denylist_survives_reload(tmp_path):
    # plik zapisany w jednym runie wyklucza w nastepnym (trwalosc na volume)
    paths = ["/x/0.wav", "/x/1.wav"]
    deny = tmp_path / "bad.txt"
    ds1 = ASRDataset(_manifest(tmp_path, paths), CharTokenizer(), bad_list_path=deny)
    ds1._mark_bad("/x/1.wav")
    ds2 = ASRDataset(_manifest(tmp_path, paths), CharTokenizer(), bad_list_path=deny)
    assert [e["audio_path"] for e in ds2.entries] == ["/x/0.wav"]
