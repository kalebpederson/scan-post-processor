"""OCR word recall (R1/R3 method): words OCR'd (conf >= 90) from the cleaned reference page must be found again in the OCR of the
render of the encoded PDF (multiset recall, psm 3, tesseract eng, one thread per process)."""
from __future__ import annotations

import collections
import csv
import io
import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

TMP = Path(__file__).resolve().parent / "work" / "tmp"


def _norm(tok: str):
    t = tok.strip()
    if not re.search(r"[A-Za-z0-9]", t):
        return None
    return re.sub(r"^[^\w]+|[^\w]+$", "", t)


def ocr(rgb: np.ndarray, dpi: int = 300, psm: int = 3):
    TMP.mkdir(parents=True, exist_ok=True)
    p = TMP / f"ocr{os.getpid()}.png"
    cv2.imwrite(str(p), cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    out = subprocess.run(["tesseract", str(p), "stdout", "-l", "eng", "--psm", str(psm), "--oem", "1", "--dpi", str(dpi), "tsv"],
                         capture_output=True, text=True, env=env).stdout
    p.unlink(missing_ok=True)
    rows = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r["level"] != "5":
            continue
        t = _norm(r["text"] or "")
        if t:
            rows.append((t, float(r["conf"])))
    return rows


_ref_cache: dict = {}


def words(rgb: np.ndarray, min_conf: float = 90.0) -> list[str]:
    """Reference words: the cleaned page's own OCR at conf >= min_conf."""
    return [t for t, c in ocr(rgb) if c >= min_conf]


def hyp_words(rgb: np.ndarray) -> list[str]:
    return [t for t, _ in ocr(rgb)]


def recall(ref_words: list[str], hyp: list[str]) -> float:
    rc, hc = collections.Counter(ref_words), collections.Counter(hyp)
    return sum(min(rc[t], hc[t]) for t in rc) / max(1, len(ref_words))
