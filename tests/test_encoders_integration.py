"""Full CLI with --encoder auto on corpus-built PDFs: size per page, both renderers, PDF validity, report scheme."""
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import img2pdf
import numpy as np
import pikepdf
import pytest

from test_encoders import render
from test_integration import WRAPPER, run_cli

PAGES = ["pair01_front.jpg", "single03.jpg", "image02.jpg", "single09.jpg"]     # text+colour / colour / gray art / highlighter

pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("pdftoppm") is None or shutil.which("mutool") is None,
                                                   reason="renderers missing")]


@pytest.fixture(scope="module")
def scan_pdf(corpus_dir, tmp_path_factory):
    files = [corpus_dir / n for n in PAGES]
    if not all(f.exists() for f in files):
        pytest.skip("corpus images missing")
    p = tmp_path_factory.mktemp("encinteg") / "Enc Scan.pdf"
    p.write_bytes(img2pdf.convert([str(f) for f in files]))
    return p


@pytest.mark.parametrize("encoder", ["auto", "mrc", "jpeg", "jpeg200"])
def test_cli_encoder(scan_pdf, tmp_path, encoder):
    work = tmp_path / "Enc Scan.pdf"
    shutil.copy(scan_pdf, work)
    rep = tmp_path / "rep.json"
    r = run_cli(work, f"--jobs 4 --foreground --keep-blank --encoder {encoder} --report {rep}")
    assert r.returncode == 0, r.stderr
    with pikepdf.open(work) as pdf:
        assert pdf.check_pdf_syntax() == []
        n = len(pdf.pages)
    assert n == len(PAGES)
    per_page = work.stat().st_size / n
    limit = {"auto": 200e3, "mrc": 200e3, "jpeg": 300e3, "jpeg200": 500e3}[encoder]
    assert per_page < limit, per_page
    pages = json.loads(rep.read_text())["files"][0]["pages"]
    assert all(p["scheme"] in ("e3", "e3x", "e1", "blank") and p["bytes"] > 0 for p in pages)
    if encoder == "auto":
        assert pages[3]["scheme"] == "e1"                    # highlighter page
        assert pages[2]["scheme"] == "e3"                    # gray art page
        assert all(p["bytes"] < 200e3 for p in pages)
    orig = tmp_path / ".originals" / "Enc Scan.pdf"
    for i in range(1, n + 1):
        a = render(orig, "pdftoppm", 100, i)
        for rr in ("pdftoppm", "mutool"):
            b = render(work, rr, 100, i)
            assert b.shape[:2] == a.shape[:2]
            assert float(np.abs(a.astype(np.int16) - b).mean()) < 25, (i, rr)       # same page (paper cleaned, so not tiny)
            assert (b.min(axis=2) >= 245).mean() > 0.5                                 # white paper


def test_dpi_only_affects_jpeg_encoders(scan_pdf, tmp_path):
    outs = {}
    for enc, dpi in (("auto", 100), ("auto", 250), ("jpeg", 100), ("jpeg", 250)):
        work = tmp_path / f"{enc}{dpi}.pdf"
        shutil.copy(scan_pdf, work)
        r = run_cli(work, f"--jobs 4 --foreground --keep-blank --no-backup --encoder {enc} --dpi {dpi}")
        assert r.returncode == 0, r.stderr
        outs[(enc, dpi)] = work.read_bytes()
    assert outs[("auto", 100)] == outs[("auto", 250)] or abs(len(outs[("auto", 100)]) - len(outs[("auto", 250)])) < 2000
    assert len(outs[("jpeg", 100)]) < len(outs[("jpeg", 250)]) * 0.7
