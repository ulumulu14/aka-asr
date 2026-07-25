"""Strumieniowa ekstrakcja tar.gz (cv2json/hf2json): pliki laduja na miejscu,
path traversal (../evil) rzuca. Regresja na pulapke getmembers()+extractall()
(2x dekompresja, cicha ekstrakcja wygladajaca jak zwis na network volume)."""

import io
import tarfile

import pytest

from utils.datasets_conversion.cv2json import safe_extract_tar_gz as cv_extract
from utils.datasets_conversion.hf2json import safe_extract_tar_gz as hf_extract


def _make_tar(path, evil=False):
    with tarfile.open(path, "w:gz") as t:
        for name in ("d/a.txt", "d/sub/b.txt"):
            data = b"ok"
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        if evil:
            info = tarfile.TarInfo("../evil.txt")
            info.size = 2
            t.addfile(info, io.BytesIO(b"xx"))


@pytest.mark.parametrize("extract", [cv_extract, hf_extract], ids=["cv2json", "hf2json"])
def test_extracts_files_and_subdirs(tmp_path, extract):
    tar = tmp_path / "t.tar.gz"
    _make_tar(tar)
    out = tmp_path / "out"
    extract(tar, out)
    got = sorted(p.relative_to(out).as_posix() for p in out.rglob("*.txt"))
    assert got == ["d/a.txt", "d/sub/b.txt"]
    assert (out / "d/a.txt").read_bytes() == b"ok"


@pytest.mark.parametrize("extract", [cv_extract, hf_extract], ids=["cv2json", "hf2json"])
def test_path_traversal_blocked(tmp_path, extract):
    tar = tmp_path / "evil.tar.gz"
    _make_tar(tar, evil=True)
    with pytest.raises(RuntimeError, match="Niebezpieczna"):
        extract(tar, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()  # zly wpis nigdy nie wyladowal
