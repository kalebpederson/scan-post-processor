"""End-to-end: simple-scan-like PDFs built from research/corpus pages, run through the real CLI wrapper."""
import hashlib
import shutil
import sys
import subprocess
from pathlib import Path

import cv2
import img2pdf
import numpy as np
import pikepdf
import pytest

ROOT = Path(__file__).resolve().parent.parent
WRAPPER = ROOT / "bin" / "scan-postprocess"
PAGES = ["pair01_front.jpg", "pair01_back.jpg", "single03.jpg", "image02.jpg"]

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm missing")]


@pytest.fixture(scope="module")
def scan_pdf(corpus_dir, tmp_path_factory):
    files = [corpus_dir / n for n in PAGES]
    if not all(f.exists() for f in files):
        pytest.skip("corpus images missing")
    d = tmp_path_factory.mktemp("integ")
    p = d / "Test Scan.pdf"
    p.write_bytes(img2pdf.convert([str(f) for f in files]))     # one DCT image per page, dpi from the JPEGs
    return p


def run_cli(pdf, args="--jobs 4 --foreground", env_dir=None):
    import os
    env = dict(os.environ, SCANPP_PYTHON=sys.executable, SCANPP_LOG_DIR=str(Path(pdf).parent / "_log"))
    return subprocess.run([str(WRAPPER), "application/pdf", "false", str(pdf), args], env=env,
                          capture_output=True, text=True, timeout=300)


def render(pdf, page, dpi=100):
    out = Path(pdf).parent / f"r{page}"
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", "-singlefile",
                    str(pdf), str(out)], check=True, capture_output=True)
    return cv2.imread(str(out) + ".png")


def test_cli_end_to_end(scan_pdf, tmp_path):
    work = tmp_path / "Test Scan.pdf"
    shutil.copy(scan_pdf, work)
    orig_sha = hashlib.sha256(work.read_bytes()).hexdigest()
    orig_size = work.stat().st_size
    with pikepdf.open(work) as pdf:
        boxes = [[float(v) for v in p.MediaBox] for p in pdf.pages]

    r = run_cli(work)
    assert r.returncode == 0, r.stderr

    with pikepdf.open(work) as pdf:
        kept = len(pdf.pages)
        assert 1 <= kept <= len(PAGES)
        out_boxes = [[float(v) for v in p.MediaBox] for p in pdf.pages]
        assert str(pdf.docinfo["/Title"]) == "Test Scan"
    assert kept >= 3                                                   # only genuinely blank backs may drop
    # MediaBoxes preserved (in order, allowing dropped blanks)
    it = iter(boxes)
    assert all(any(b == pytest.approx(c, abs=0.01) for c in it) for b in out_boxes)
    assert work.stat().st_size < orig_size

    backup = tmp_path / ".originals" / "Test Scan.pdf"
    assert backup.exists() and hashlib.sha256(backup.read_bytes()).hexdigest() == orig_sha

    for i in range(1, kept + 1):
        img = render(work, i)
        assert img is not None and img.shape[0] > 100


def test_golden_paper_is_white(scan_pdf, tmp_path):
    work = tmp_path / "g.pdf"
    shutil.copy(scan_pdf, work)
    assert run_cli(work, "--jobs 2 --foreground").returncode == 0
    img = render(work, 1, 100)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    near_ink = cv2.dilate((gray < 200).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    paper = ~near_ink
    white = (img.min(axis=2) >= 245)[paper].mean()
    assert white >= 0.99, white


def test_digital_pdf_skipped_by_cli(tmp_path):
    from helpers import make_digital_pdf
    f = make_digital_pdf(tmp_path / "d.pdf")
    before = f.read_bytes()
    r = run_cli(f)
    assert r.returncode == 0 and f.read_bytes() == before
    assert not (tmp_path / ".originals").exists()


def test_failure_exit_code_nonzero(tmp_path):
    f = tmp_path / "bad.pdf"
    f.write_bytes(b"%PDF-1.4\nthis is not a pdf")
    r = run_cli(f)
    assert r.returncode == 1 and f.read_bytes().startswith(b"%PDF-1.4\nthis is")
