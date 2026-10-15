import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "research" / "corpus"


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANPP_LOG_DIR", str(tmp_path / "_log"))


@pytest.fixture(autouse=True)
def recorded_notifications(monkeypatch):
    """Tests must never pop up real desktop notifications: record calls instead (and tell subprocesses)."""
    monkeypatch.setenv("SCANPP_NOTIFY", "0")
    monkeypatch.delenv("SCANPP_NOTIFY_CMD", raising=False)
    calls = []

    def fake(summary, body="", urgent=False, expire_ms=6000):
        calls.append({"summary": summary, "body": body, "urgent": urgent})
        return True

    import scanpp.background, scanpp.cli, scanpp.notify
    for mod in (scanpp.notify, scanpp.cli, scanpp.background):
        monkeypatch.setattr(mod, "notify", fake)
    return calls


@pytest.fixture(scope="session")
def corpus_dir():
    if not CORPUS.is_dir():
        pytest.skip("research/corpus not present")
    return CORPUS
