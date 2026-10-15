import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pikepdf
import pytest

from helpers import make_digital_pdf, make_scan_pdf, text_page
from scanpp import background, cli, safety

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "bin" / "scan-postprocess"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def n_pages(p):
    with pikepdf.open(p) as pdf:
        return len(pdf.pages)


@pytest.fixture
def notes(tmp_path, monkeypatch):
    """Fake notify-send that records its arguments (one line per call)."""
    out = tmp_path / "_notes.txt"
    script = tmp_path / "_fake-notify"
    script.write_text(f'#!/bin/bash\nprintf "%s|" "$@" >> "{out}"; echo >> "{out}"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("SCANPP_NOTIFY_CMD", str(script))
    return out


def wait_for(pred, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.1)
    return False


def simulate_simple_scan(path, args="--jobs 2", extra_env=None):
    env = dict(os.environ, SCANPP_PYTHON=sys.executable, **(extra_env or {}))
    t0 = time.perf_counter()
    p = subprocess.Popen([str(WRAPPER), "application/pdf", "false", str(path), args], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(timeout=30)      # waits for exit AND EOF on both pipes, like simple-scan
    return p.returncode, time.perf_counter() - t0, out, err


@pytest.mark.slow
def test_background_returns_fast_and_completes(tmp_path, notes):
    pages = [text_page(seed=i) for i in range(14)]
    f = make_scan_pdf(tmp_path / "bg.pdf", pages)
    before = sha(f)
    rep = tmp_path / "rep.json"
    rc, dt, out, err = simulate_simple_scan(f, f"--jobs 1 --report {rep}")
    assert rc == 0 and dt < 2.0, dt
    assert sha(f) == before                                     # processing has not finished yet
    assert (tmp_path / ".originals" / "bg.pdf").exists()        # but the backup already exists
    assert wait_for(lambda: notes.exists() and notes.read_text().strip())
    assert sha(f) != before and n_pages(f) == 14
    assert sha(tmp_path / ".originals" / "bg.pdf") == before
    data = json.loads(rep.read_text())
    assert len(data["files"][0]["pages"]) == 14
    assert "14 pages (0 blank dropped)" in notes.read_text() and "scanpp" in notes.read_text()
    log = (tmp_path / "_log" / "scanpp.log").read_text()
    assert "job=" in log and "outcome=processed" in log


@pytest.mark.slow
def test_worker_does_not_inherit_pipes(tmp_path, notes):
    """communicate() returning (EOF on both pipes) while the worker still runs proves the fds were not inherited."""
    f = make_scan_pdf(tmp_path / "p.pdf", [text_page(seed=i) for i in range(14)])
    rc, dt, out, err = simulate_simple_scan(f, "--jobs 1")
    assert rc == 0 and dt < 2.0
    assert wait_for(lambda: notes.exists())


def test_no_notify_flag(tmp_path, notes):
    f = make_scan_pdf(tmp_path / "n.pdf", [text_page()])
    before = sha(f)
    rc, dt, *_ = simulate_simple_scan(f, "--jobs 1 --no-notify")
    assert rc == 0
    assert wait_for(lambda: sha(f) != before)
    time.sleep(0.5)
    assert not notes.exists()


def test_jobs_serialize_on_lock(tmp_path, notes):
    f = make_scan_pdf(tmp_path / "q.pdf", [text_page()])
    before = sha(f)
    lock = background.lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        rc, dt, *_ = simulate_simple_scan(f, "--jobs 1", {"SCANPP_LOG_DIR": os.environ["SCANPP_LOG_DIR"]})
        assert rc == 0
        time.sleep(2.0)
        assert sha(f) == before                                  # worker is waiting for the lock
    assert wait_for(lambda: sha(f) != before)


def _launch_only(path, monkeypatch, extra=""):
    monkeypatch.setattr(background.subprocess, "Popen", lambda *a, **k: None)
    ns, keep = cli.parse_args(["application/pdf", "false", str(path), extra or "--jobs 1"])
    assert background.launch(str(path), cli.options_from(ns), keep) == 0
    jobs = sorted((safety.log_dir() / "jobs").glob("*.json"))
    return jobs[-1]


def test_target_changed_writes_sidecar(tmp_path, monkeypatch):
    f = make_scan_pdf(tmp_path / "c.pdf", [text_page(), text_page(seed=2)])
    orig = sha(f)
    job = _launch_only(f, monkeypatch)
    f2 = make_scan_pdf(tmp_path / "c.pdf", [text_page(seed=9)])       # user re-saved the file meanwhile
    resaved = sha(f2)
    sent = []
    monkeypatch.setattr(background, "notify", lambda *a, **k: sent.append((a, k)))
    assert background.run_worker(str(job)) == 0
    assert sha(f) == resaved                                           # not overwritten
    side = tmp_path / "c.scanpp.pdf"
    assert side.exists() and n_pages(side) == 2                        # processed from the backup (2 pages)
    assert sha(tmp_path / ".originals" / "c.pdf") == orig
    assert sent and sent[0][1]["urgent"] and "changed" in sent[0][0][1] and "scanpp.log" in sent[0][0][1]
    assert not job.exists()


def test_target_deleted_discards(tmp_path, monkeypatch):
    f = make_scan_pdf(tmp_path / "d.pdf", [text_page()])
    job = _launch_only(f, monkeypatch)
    f.unlink()
    sent = []
    monkeypatch.setattr(background, "notify", lambda *a, **k: sent.append((a, k)))
    assert background.run_worker(str(job)) == 0
    assert [p.name for p in tmp_path.iterdir() if not p.name.startswith("_")] == [".originals"]
    assert sent and sent[0][1]["urgent"] and "deleted" in sent[0][0][1]


def test_worker_failure_notifies_and_keeps_original(tmp_path, monkeypatch):
    f = make_scan_pdf(tmp_path / "e.pdf", [text_page()])
    before = sha(f)
    job = _launch_only(f, monkeypatch)
    from scanpp import pipeline
    monkeypatch.setattr(pipeline, "clean_page", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    sent = []
    monkeypatch.setattr(background, "notify", lambda *a, **k: sent.append((a, k)))
    assert background.run_worker(str(job)) == 1
    assert sha(f) == before
    assert sent and "failed" in sent[0][0][0] and sent[0][1]["urgent"]
    assert "boom" in (tmp_path / "_log" / "scanpp.log").read_text()


def test_notify_missing_is_silent(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANPP_NOTIFY_CMD", "/nonexistent/notify-send")
    from scanpp.notify import send_notification
    assert send_notification("x", "y") is False


def test_real_notify_refused_under_pytest(tmp_path, monkeypatch):
    marker = tmp_path / "called"
    script = tmp_path / "ns"
    script.write_text(f"#!/bin/bash\ntouch {marker}\n")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:" + os.environ["PATH"])
    monkeypatch.delenv("SCANPP_NOTIFY_CMD", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "x")
    from scanpp.notify import send_notification
    script.rename(tmp_path / "notify-send")
    assert send_notification("x") is False and not marker.exists()
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    monkeypatch.setenv("SCANPP_NOTIFY", "0")
    assert send_notification("x") is False and not marker.exists()


def test_mode_selection(tmp_path):
    def mode(args, protocol, name="a.pdf"):
        ns, _ = cli.parse_args(args)
        return cli.should_background(ns, protocol)
    assert mode(["application/pdf", "true", "/x/a.pdf"], True)
    assert not mode(["application/pdf", "true", "/x/a.pdf", "--foreground"], True)
    assert not mode(["/x/a.pdf"], False)
    assert mode(["--background", "/x/a.pdf"], False)
    assert not mode(["--background", "/x/a.pdf", "/x/b.pdf"], False)


@pytest.mark.parametrize("p,expect", [("/tmp/simple-scan-AB12CD/scan.pdf", True), ("/var/tmp/simple-scan-a1B2c3/x.jpg", True),
                                      ("/tmp/simple-scan-AB12C/scan.pdf", False), ("/tmp/simple-scan-AB12CDE/s.pdf", False),
                                      ("/home/u/Documents/scan.pdf", False)])
def test_email_dir_detection(p, expect):
    assert background.is_email_tmp(p) is expect


def test_email_dir_forces_foreground(tmp_path):
    d = tmp_path / "simple-scan-Ab3Xy9"
    d.mkdir()
    f = make_scan_pdf(d / "scan.pdf", [text_page(), text_page(seed=3)])
    before = sha(f)
    assert cli.main(["application/pdf", "false", str(f), "--jobs 1"]) == 0     # protocol, no --foreground
    assert sha(f) != before and n_pages(f) == 2                                 # already processed on return


def test_background_skips_digital_pdf(tmp_path, notes):
    f = make_digital_pdf(tmp_path / "dig.pdf")
    before = sha(f)
    rc, dt, *_ = simulate_simple_scan(f)
    assert rc == 0 and sha(f) == before and not (tmp_path / ".originals").exists()
    time.sleep(0.3)
    assert not notes.exists()


def test_foreground_failure_in_protocol_mode_notifies(tmp_path, recorded_notifications):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4\nnot a pdf")
    assert cli.main(["application/pdf", "false", str(f), "--foreground"]) == 1
    assert len(recorded_notifications) == 1 and "failed" in recorded_notifications[0]["summary"]


def test_plain_cli_never_notifies(tmp_path, recorded_notifications):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4\nnot a pdf")
    assert cli.main([str(f)]) == 1
    assert cli.main([str(tmp_path / "missing.pdf")]) == 1
    good = make_scan_pdf(tmp_path / "ok.pdf", [text_page()])
    assert cli.main([str(good), "--jobs", "1"]) == 0
    assert recorded_notifications == []


def test_plain_cli_background_job_never_notifies(tmp_path, monkeypatch, recorded_notifications):
    f = make_scan_pdf(tmp_path / "pb.pdf", [text_page()])
    monkeypatch.setattr(background.subprocess, "Popen", lambda *a, **k: None)
    ns, keep = cli.parse_args(["--background", "--jobs", "1", str(f)])
    background.launch(str(f), cli.options_from(ns), keep)
    job = sorted((safety.log_dir() / "jobs").glob("*.json"))[-1]
    assert background.run_worker(str(job)) == 0
    assert recorded_notifications == []
