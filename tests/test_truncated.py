"""Damaged image data (short/over-long Flate, truncated JPEG) must not fail the document: pad/truncate + WARNING."""
import hashlib
import os
import zlib
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pytest
from pikepdf import Name

from helpers import jpeg_bytes, make_scan_pdf, text_page
from scanpp import cli, pdfio

W, H = 640, 800


def _run(src, tmp_path, capsys=None):
    out = tmp_path / "out.pdf"
    rc = cli.main(["--foreground", "--no-backup", "--no-duplex" if False else "--duplex", "off", "-o", str(out), str(src)])
    assert rc == 0
    log = (Path(os.environ["SCANPP_LOG_DIR"]) / "scanpp.log").read_text()
    return out, log


def _pages(n=3):
    return [text_page(w=W, h=H, seed=i) for i in range(n)]


def _damage_page(path, idx, fn):
    with pikepdf.open(str(path), allow_overwriting_input=True) as pdf:
        st = list(pdf.pages[idx].resources.XObject.values())[0]
        fn(st)
        pdf.save(str(path))


def _recompress(st, new):
    st.write(zlib.compress(new), filter=Name.FlateDecode)


@pytest.mark.parametrize("kind,ncomp", [("flate", 3), ("flate_gray", 1)])
def test_short_flate_is_padded(tmp_path, kind, ncomp):
    src = make_scan_pdf(tmp_path / "in.pdf", _pages(), kind=kind)
    cut = 2 * W * ncomp + 17                      # two rows and a partial one missing
    _damage_page(src, 1, lambda st: _recompress(st, st.read_bytes()[:-cut]))
    out, log = _run(src, tmp_path)
    with pikepdf.open(str(out)) as p:
        assert len(p.pages) == 3
    assert "image data is short" in log and "in.pdf: page 2" in log and "padded 3 of 800 rows" in log
    with pikepdf.open(str(src)) as p:
        pi = pdfio.extract_page_input(p.pages[1], 1)
    arr = np.frombuffer(pi.data, np.uint8).reshape(H, W, 3)
    assert (arr[-1] == arr[-4]).all() and (arr[-2] == arr[-4]).all()   # padded rows replicate the last complete row


def test_partial_row_completed_from_row_above():
    data = bytes(range(10)) * 3                       # 3 rows of 10
    fixed, rows = pdfio._complete_raw(data[:25], 10, 4)
    assert rows == 2 and len(fixed) == 40
    assert fixed[20:30] == data[20:25] + data[15:20]  # partial row 2 completed from row 1
    assert fixed[30:] == fixed[20:30]


def test_overlong_flate_is_truncated(tmp_path):
    src = make_scan_pdf(tmp_path / "in.pdf", _pages(), kind="flate")
    _damage_page(src, 0, lambda st: _recompress(st, st.read_bytes() + b"\x07" * 5000))
    out, log = _run(src, tmp_path)
    with pikepdf.open(str(out)) as p:
        assert len(p.pages) == 3
    assert "image data is too long" in log


def test_truncated_jpeg_is_padded(tmp_path):
    src = make_scan_pdf(tmp_path / "in.pdf", _pages(), kind="jpeg")
    _damage_page(src, 2, lambda st: st.write(st.read_raw_bytes()[: len(st.read_raw_bytes()) * 6 // 10], filter=Name.DCTDecode))
    out, log = _run(src, tmp_path)
    with pikepdf.open(str(out)) as p:
        assert len(p.pages) == 3
    assert "JPEG data is truncated" in log and "in.pdf: page 3" in log
    with pikepdf.open(str(src)) as p:
        pi = pdfio.extract_page_input(p.pages[2], 2)
    assert pi.kind == "raw" and pi.size == (W, H)


def test_normal_files_unchanged(tmp_path):
    """Intact images keep the fast paths: jpeg bytes passed through untouched, flate pixels equal the source."""
    pages = _pages(2)
    j = make_scan_pdf(tmp_path / "j.pdf", pages, kind="jpeg")
    f = make_scan_pdf(tmp_path / "f.pdf", pages, kind="flate")
    with pikepdf.open(str(j)) as pdf:
        st = list(pdf.pages[0].resources.XObject.values())[0]
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
        assert pi.kind == "jpeg" and hashlib.sha256(pi.data).digest() == hashlib.sha256(bytes(st.read_raw_bytes())).digest()
    with pikepdf.open(str(f)) as pdf:
        pi = pdfio.extract_page_input(pdf.pages[1], 1)
        assert pi.kind == "raw" and pi.data == pages[1].tobytes()
    out, log = _run(f, tmp_path)
    assert "short" not in log and "truncated" not in log and "too long" not in log
