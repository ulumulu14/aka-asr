"""HfSync: no-op bez repo_id, poprawne path_in_repo (podkatalog runu), bledy HF
nie propaguja. Bez realnego HF (mock HfApi)."""

from utils.hf_sync import HfSync


def test_disabled_without_repo(tmp_path):
    hf = HfSync(None, run_subdir="conformer_big/2026")
    assert hf.enabled is False
    # wszystkie metody to no-op, nic nie rzucaja
    hf.push_metrics(tmp_path, tmp_path / "train.log")
    hf.push_best(tmp_path / "best.pt")
    hf.finalize()


class _FakeApi:
    """Zbiera wywolania upload zamiast gadac z HF."""

    def __init__(self, *a, **k):
        self.calls = []

    def create_repo(self, repo_id, *a, **k):
        # HF rozwiazuje repo_id bez namespace -> <owner>/name (jak realne create_repo)
        full = repo_id if "/" in repo_id else f"owner/{repo_id}"
        return type("RepoUrl", (), {"repo_id": full})()

    def upload_folder(self, **k):
        self.calls.append(("folder", k["path_in_repo"]))
        return _DoneFuture()

    def upload_file(self, **k):
        self.calls.append(("file", k["path_in_repo"]))
        return _DoneFuture()


class _DoneFuture:
    def result(self, timeout=None):
        return None


def _enabled_sync(monkeypatch, run_subdir="conformer_big/2026-07-25"):
    import huggingface_hub

    fake = _FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: fake)
    hf = HfSync("user/aka-asr-runs", run_subdir=run_subdir)
    return hf, fake


def test_repo_id_without_namespace_resolved(monkeypatch):
    # "aka-asr-v1" (bez namespace) -> HF rozwiazuje na <owner>/aka-asr-v1
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _FakeApi())
    hf = HfSync("aka-asr-v1", run_subdir="conformer_big/x")
    assert hf.repo_id == "owner/aka-asr-v1"


def test_paths_are_run_subdir(monkeypatch, tmp_path):
    (tmp_path / "tb").mkdir()
    (tmp_path / "train.log").write_text("x", encoding="utf-8")
    (tmp_path / "best.pt").write_bytes(b"weights")
    hf, fake = _enabled_sync(monkeypatch)
    assert hf.enabled is True
    hf.push_metrics(tmp_path / "tb", tmp_path / "train.log")
    hf.push_best(tmp_path / "best.pt")
    paths = [p for _, p in fake.calls]
    assert "conformer_big/2026-07-25/tb" in paths
    assert "conformer_big/2026-07-25/train.log" in paths
    assert "conformer_big/2026-07-25/best.pt" in paths
    hf.finalize()  # DoneFuture -> nie wisi


def test_push_best_skips_missing_file(monkeypatch, tmp_path):
    hf, fake = _enabled_sync(monkeypatch)
    hf.push_best(tmp_path / "nie_ma.pt")  # brak pliku -> nie uploaduje
    assert fake.calls == []


def test_hf_error_does_not_propagate(monkeypatch, tmp_path):
    (tmp_path / "best.pt").write_bytes(b"w")
    hf, fake = _enabled_sync(monkeypatch)

    def boom(**k):
        raise RuntimeError("HF down")

    fake.upload_file = boom
    hf.push_best(tmp_path / "best.pt")  # blad HF nie moze ubic treningu
