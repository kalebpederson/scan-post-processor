"""Safety helpers: logging, backups of originals, atomic replacement."""
from __future__ import annotations

import logging
import logging.handlers
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

log = logging.getLogger("scanpp")


def log_dir() -> Path:
    env = os.environ.get("SCANPP_LOG_DIR")
    return Path(env) if env else Path.home() / ".cache" / "scanpp"


def setup_logging(verbose: bool = False, job_id: str | None = None, stderr: bool = True) -> None:
    """Log to stderr and to a rotating file ~/.cache/scanpp/scanpp.log (never fatal if the file is unwritable)."""
    log.setLevel(logging.DEBUG)
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    job = f" job={job_id}" if job_id else ""
    fmt = logging.Formatter(f"%(asctime)s %(levelname)s [%(process)d{job}] %(message)s")
    if stderr:
        sh = logging.StreamHandler(sys.stderr)
        sh.setLevel(logging.DEBUG if verbose else logging.INFO)
        sh.setFormatter(fmt)
        log.addHandler(sh)
    try:
        d = log_dir()
        d.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(d / "scanpp.log", maxBytes=1_000_000, backupCount=3,
                                                  encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except OSError as e:  # pragma: no cover
        log.warning("cannot open log file: %s", e)
    log.propagate = False


def backup_path_for(path: Path, backup_dir: Path | None = None) -> Path:
    """Where the backup of `path` goes: <backup_dir or dir/.originals>/<name>, with a timestamp (and counter)
    suffix inserted before the extension when that name is already taken."""
    path = Path(path)
    d = Path(backup_dir) if backup_dir else path.parent / ".originals"
    cand = d / path.name
    if not cand.exists():
        return cand
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cand = d / f"{path.stem}.{stamp}{path.suffix}"
    n = 1
    while cand.exists():
        cand = d / f"{path.stem}.{stamp}-{n}{path.suffix}"
        n += 1
    return cand


def backup_original(path: Path, backup_dir: Path | None = None) -> Path:
    """Copy `path` to its backup location and verify the copy byte-for-byte (size + content). Raises on failure."""
    path = Path(path)
    dest = backup_path_for(path, backup_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    if dest.stat().st_size != path.stat().st_size:
        raise OSError(f"backup size mismatch: {dest}")
    with open(path, "rb") as a, open(dest, "rb") as b:
        while True:
            x, y = a.read(1 << 20), b.read(1 << 20)
            if x != y:
                raise OSError(f"backup content mismatch: {dest}")
            if not x:
                break
    return dest


def make_temp_path(target: Path) -> Path:
    """A fresh temp file name in the same directory as `target` (so os.replace is atomic)."""
    fd, name = tempfile.mkstemp(prefix=".scanpp-", suffix=Path(target).suffix or ".tmp", dir=Path(target).parent)
    os.close(fd)
    return Path(name)


def atomic_replace(tmp: Path, target: Path) -> None:
    """Atomically move `tmp` over `target`, preserving the target's permission bits when it exists."""
    target = Path(target)
    if target.exists():
        try:
            shutil.copymode(target, tmp)
        except OSError:
            pass
    os.replace(tmp, target)


_EDGE = 1 << 20


def fingerprint(path) -> dict:
    """Cheap identity of a file: size, mtime_ns and sha256 over the first and last 1 MB."""
    import hashlib
    path = Path(path)
    st = path.stat()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(_EDGE))
        if st.st_size > 2 * _EDGE:
            f.seek(-_EDGE, os.SEEK_END)
            h.update(f.read(_EDGE))
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha": h.hexdigest()}


def fingerprint_matches(fp: dict, path) -> bool:
    try:
        return fingerprint(path) == fp
    except OSError:
        return False
