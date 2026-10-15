"""Background mode: quick foreground part (classify, backup, fingerprint, spawn) + detached, serialized worker.

simple-scan blocks until the script exits AND its stdout/stderr pipes hit EOF, so the worker is started in a new
session with stdin/stdout/stderr = /dev/null and close_fds (it logs to the log file only). Workers take an
exclusive flock so that queued scans are processed one at a time (each already uses all CPUs).
"""
from __future__ import annotations

import dataclasses
import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import safety
from .notify import notify
from .options import Options

log = logging.getLogger("scanpp")

_EMAIL_DIR = re.compile(r"^simple-scan-[A-Za-z0-9]{6}$")


def is_email_tmp(path) -> bool:
    """simple-scan's Email action saves into <tmp>/simple-scan-XXXXXX/ and attaches the file as soon as we return."""
    return bool(_EMAIL_DIR.match(Path(path).resolve().parent.name))


def lock_path() -> Path:
    return safety.log_dir() / "worker.lock"


def _jobs_dir() -> Path:
    d = safety.log_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def quick_classify(path: Path) -> tuple[str, str]:
    """Cheap pre-check: returns (kind, reason) with kind in {"pdf", "image", "skip"}."""
    import pikepdf
    from . import imageio, pdfio
    fmt = imageio.sniff_format(path)
    if fmt == "PDF":
        try:
            with pikepdf.open(str(path)) as pdf:
                ok, why = pdfio.classify_pdf(pdf)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"cannot open PDF: {e}") from e
        return ("pdf", "ok") if ok else ("skip", why)
    if fmt in imageio.SUPPORTED:
        return "image", "ok"
    return "skip", "unsupported file type"


def summary_text(res) -> str:
    dropped = res.pages_in - res.pages_out
    pages = f"{res.pages_out} page{'s' if res.pages_out != 1 else ''}"
    return (f"{Path(res.path).name}: {pages} ({dropped} blank dropped), "
            f"{res.bytes_in / 1e6:.1f} MB → {res.bytes_out / 1e6:.1f} MB")


def _log_path() -> str:
    return str(safety.log_dir() / "scanpp.log")


def launch(path: str, opts: Options, keep_original: str | None) -> int:
    """Foreground part. Returns the exit code for the script."""
    path = Path(os.path.abspath(path))
    t0 = time.perf_counter()
    if not path.is_file():
        raise FileNotFoundError(path)
    kind, why = quick_classify(path)
    if kind == "skip":
        log.info("%s: not processed (%s); leaving untouched", path, why)
        return 0
    log.info("KEEP_ORIGINAL=%s (ignored: originals are always backed up unless --no-backup)", keep_original)
    backup = None
    if opts.backup:
        bdir = Path(os.path.abspath(opts.backup_dir)) if opts.backup_dir else None
        backup = safety.backup_original(path, bdir)
        log.info("backed up %s -> %s", path, backup)
    else:
        log.warning("backup disabled: %s will be overwritten without a copy", path)
    fp = safety.fingerprint(path)
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}"
    d = dataclasses.asdict(opts)
    if keep_original is None:          # plain CLI (not simple-scan): never notify
        d["notify"] = False
    for k in ("backup_dir", "report"):
        if d.get(k):
            d[k] = os.path.abspath(d[k])
    job = {"id": job_id, "path": str(path), "source": str(backup) if backup else None, "fingerprint": fp,
           "keep_original": keep_original, "options": d}
    jf = _jobs_dir() / f"{job_id}.json"
    jf.write_text(json.dumps(job))
    subprocess.Popen([sys.executable, "-m", "scanpp", "--_worker", str(jf)],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     close_fds=True, start_new_session=True, cwd="/")
    log.info("job %s queued (%s), foreground part took %.3fs", job_id, path.name, time.perf_counter() - t0)
    return 0


def run_worker(jobfile: str) -> int:
    from .pipeline import process_file, write_report
    job = json.loads(Path(jobfile).read_text())
    safety.setup_logging(False, job_id=job["id"], stderr=False)
    try:
        os.nice(10)          # stay out of the way of the next scan's foreground part / the desktop
    except OSError:
        pass
    opts = Options(**job["options"])
    path = Path(job["path"])
    try:
        log.info("job %s: waiting for worker lock", job["id"])
        t0 = time.perf_counter()
        lock_path().parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path(), "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            log.info("job %s: lock acquired after %.1fs; processing %s", job["id"], time.perf_counter() - t0, path)
            res = process_file(path, opts, keep_original=job["keep_original"], source=job["source"],
                               backup_done=True, fingerprint=job["fingerprint"])
        log.info("job %s: outcome=%s note=%r wall=%.2fs stage_cpu=%s", job["id"], res.status, res.note, res.wall,
                 res.stage_cpu)
        if res.status == "processed":
            log.info("job %s: %s", job["id"], summary_text(res))
        if opts.report:
            write_report(opts.report, opts, [res])
        if opts.notify:
            if res.note:
                notify(f"scanpp: {path.name}", f"{res.note}. Log: {_log_path()}", urgent=True)
            elif res.status == "processed":
                notify("scanpp", summary_text(res))
        return 0
    except Exception:  # noqa: BLE001
        log.error("job %s FAILED; original left untouched\n%s", job["id"], traceback.format_exc())
        if opts.notify:
            notify(f"scanpp failed: {path.name}",
                   f"Original kept (backup in .originals). See {_log_path()}", urgent=True)
        return 1
    finally:
        Path(jobfile).unlink(missing_ok=True)
