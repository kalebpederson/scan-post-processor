"""--quality N (1..10): settings ladder, CLI validation, and synthetic end-to-end checks (bytes, allowed filters, re-processing guard).

Level 5 must reproduce the historical encoders exactly; the ladder is documented in README "Quality" and measured in
research/r8_quality.
"""
from __future__ import annotations

import dataclasses
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pytest

from scanpp import encode, mrc, pdfio
from scanpp.cli import build_parser, options_from, parse_args
from scanpp.encode import QUALITY_LADDER, QualitySettings, encode_page, quality_settings
from scanpp.options import Options

from helpers import make_scan_pdf
from test_encoders import synth_page
from test_integration import run_cli

ALLOWED_FILTERS = {"/DCTDecode", "/JBIG2Decode", "/CCITTFaxDecode", "/FlateDecode"}   # baseline JPEG, JBIG2 generic, G4, lossless
needs_jbig2 = pytest.mark.skipif(mrc.jbig2_path() is None, reason="jbig2 missing")


# ----------------------------------------------------------------------------- unit: the ladder
def test_level_5_is_the_default_settings_object():
    assert quality_settings(5) == QualitySettings() == QUALITY_LADDER[5]
    assert encode.DEFAULT_QUALITY == 5 and Options().quality == 5
    qs = quality_settings(5)                      # ... and these are the historical constants
    assert (qs.col_ppi, qs.col_q, qs.mask_ppi, qs.e1_ppi, qs.e1_q, qs.full_page) == (100, 40, 300, 150, 55, False)
    assert encode.MRC_COL_PPI == 100 and encode.MRC_COL_Q == 40 and encode.JPEG_Q_E1 == 55 and encode.MRC_MASK_PPI == 300


def test_ladder_covers_1_to_10_and_is_monotone():
    assert sorted(QUALITY_LADDER) == list(range(1, 11))
    levels = [QUALITY_LADDER[n] for n in range(1, 11)]
    assert [q.level for q in levels] == list(range(1, 11))
    for a, b in zip(levels, levels[1:]):
        assert (b.col_ppi, b.col_q) >= (a.col_ppi, a.col_q), (a, b)       # colour layer: resolution first, then JPEG quality
        assert b.col_ppi >= a.col_ppi, (a, b)
        assert b.e1_ppi >= a.e1_ppi and b.e1_q >= a.e1_q, (a, b)          # every knob individually non-decreasing
        assert b.mask_ppi >= a.mask_ppi and b.full_page >= a.full_page
        assert b != a                                                      # every step changes something
    assert all(q.e1_ppi >= 150 for q in levels)                            # R1: 150 ppi is the legibility floor for text pages
    assert all(q.mask_ppi >= 300 for q in levels)                          # R3: lower-resolution masks lose light text
    assert levels[0].col_ppi < 100 or levels[0].col_q < 40                 # level 1 really is smaller than 5
    assert levels[-1].full_page and levels[-1].e1_ppi == 300 and levels[-1].e1_q >= 90      # level 10: near-lossless JPEG


def test_quality_settings_accepts_objects_and_rejects_garbage():
    qs = QualitySettings(col_ppi=123)
    assert quality_settings(qs) is qs
    for bad in (0, 11, -1, "x", None, 5.5 + 1j):
        with pytest.raises(ValueError):
            quality_settings(bad)
    with pytest.raises(ValueError):
        encode_page(np.full((60, 60, 3), 255, np.uint8), 300, (14, 14), "auto", None, 11)


# ----------------------------------------------------------------------------- unit: command line
@pytest.mark.parametrize("bad", ["0", "11", "-3", "5.5", "abc", "", "100"])
def test_quality_rejects_out_of_range_and_garbage(bad):
    with pytest.raises(SystemExit):
        parse_args([f"--quality={bad}", "a.pdf"])


@pytest.mark.parametrize("n", range(1, 11))
def test_quality_accepts_1_to_10(n):
    ns, _ = parse_args(["--quality", str(n), "a.pdf"])
    assert options_from(ns).quality == n


def test_quality_default_and_simple_scan_args():
    ns, _ = parse_args(["a.pdf"])
    assert options_from(ns).quality == 5
    ns, keep = parse_args(["application/pdf", "true", "/x/a.pdf", "--quality 8 --paper keep"])
    o = options_from(ns)
    assert o.quality == 8 and o.paper == "keep"
    with pytest.raises(SystemExit):
        parse_args(["application/pdf", "true", "/x/a.pdf", "--quality 11"])


def test_quality_help_says_encoding_only_and_points_to_paper_keep():
    h = " ".join(build_parser().format_help().split())
    i = h.index("--quality N output image quality")
    text = h[i:i + 1400]
    assert "ENCODED only" in text and "--paper keep" in text and "--dpi" in text and "default 5" in text


# ----------------------------------------------------------------------------- encoders on a synthetic colour page
def colour_page():
    return synth_page(1200, 1500, colors=((0, 0, 0), (200, 20, 20), (20, 40, 200), (20, 140, 40)))


def image_filters(pdf_path):
    fs = set()
    with pikepdf.open(pdf_path) as pdf:
        def walk(o):
            if o.get("/Subtype") == "/Image":
                f = o.get("/Filter")
                fs.update(str(x) for x in (f if isinstance(f, pikepdf.Array) else [f] if f is not None else []))
                for k in ("/Mask", "/SMask"):
                    if k in o:
                        walk(o[k])
        for page in pdf.pages:
            for v in page.Resources.get("/XObject", {}).values():
                walk(v)
    return fs


def jpeg_sampling(data: bytes):
    """Component sampling factors of a baseline JPEG ((1,1) for all components = 4:4:4) and whether it is baseline sequential."""
    i = 2
    while i < len(data):
        assert data[i] == 0xFF
        m = data[i + 1]
        n = int.from_bytes(data[i + 2:i + 4], "big")
        if m in (0xC0, 0xC1, 0xC2):
            nc = data[i + 9]
            fac = [(data[i + 11 + 3 * k] >> 4, data[i + 11 + 3 * k] & 15) for k in range(nc)]
            return m, fac
        i += 2 + n
    raise AssertionError("no SOF")


def test_level_5_equals_default_encoding():
    img = colour_page()
    a = encode_page(img, 300, (288, 360), "auto")
    b = encode_page(img, 300, (288, 360), "auto", None, 5)
    assert {k: (v.data, v.filter) for k, v in a.xobjects.items()} == {k: (v.data, v.filter) for k, v in b.xobjects.items()}
    assert a.content == b.content
    for enc in ("mrc", "jpeg", "jpeg200"):
        x, y = encode_page(img, 300, (288, 360), enc), encode_page(img, 300, (288, 360), enc, None, 5)
        assert [v.data for v in x.xobjects.values()] == [v.data for v in y.xobjects.values()]


@needs_jbig2
def test_bytes_non_decreasing_1_5_10_on_colour_page():
    img = colour_page()
    nb = {n: encode_page(img, 300, (288, 360), "auto", None, n).nbytes for n in range(1, 11)}
    assert nb[1] < nb[5] < nb[10], nb
    assert all(a <= b for a, b in zip(list(nb.values()), list(nb.values())[1:])), nb


@needs_jbig2
@pytest.mark.parametrize("encoder", ["auto", "mrc", "jpeg", "jpeg200"])
def test_every_level_uses_only_allowed_filters_and_444_baseline_jpeg(tmp_path, encoder):
    img = colour_page()
    sizes = []
    for n in range(1, 11):
        p = tmp_path / f"{encoder}{n}.pdf"
        eps = []
        h, w = img.shape[:2]
        ep = encode_page(img, 300, (w * 72 / 300, h * 72 / 300), encoder, None, n)
        ep.media_box = (0.0, 0.0, w * 72 / 300, h * 72 / 300)
        pdfio.write_pdf([ep], p)
        with pikepdf.open(p) as pdf:
            assert pdf.check_pdf_syntax() == []
        assert image_filters(p) <= ALLOWED_FILTERS, (n, image_filters(p))
        for spec in ep.xobjects.values():
            if spec.filter == "DCTDecode":
                sof, fac = jpeg_sampling(spec.data)
                assert sof == 0xC0                                   # baseline sequential, not progressive
                assert all(f == (1, 1) for f in fac) or len(fac) == 1, (n, fac)      # never chroma-subsampled
        sizes.append(ep.nbytes)
    assert sizes == sorted(sizes), sizes


def test_gray_page_every_level_is_valid_and_monotone(tmp_path):
    g = np.full((1500, 1200, 3), 255, np.uint8)
    for k, y in enumerate(range(100, 1400, 90)):
        cv2.putText(g, "Algebra x=4 triangle ABC", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 3, cv2.LINE_AA)
    sizes = []
    for n in (1, 5, 10):
        ep = encode_page(g, 300, (288, 360), "auto", None, n)
        assert ep.xobjects and all(s.colorspace == "DeviceGray" for s in ep.xobjects.values())
        sizes.append(ep.nbytes)
    assert sizes == sorted(sizes)


def test_explicit_dpi_overrides_jpeg_encoder_ppi_at_any_level():
    img = colour_page()
    for n in (1, 5, 10):
        e = encode_page(img, 300, (288, 360), "jpeg", 100, n)
        assert e.info["jpeg_ppi"] == 100.0 and e.xobjects["Im0"].width == 400
    assert encode_page(img, 300, (288, 360), "jpeg", None, 10).info["jpeg_ppi"] == 300.0
    assert encode_page(img, 300, (288, 360), "jpeg", None, 5).info["jpeg_ppi"] == 150.0
    assert encode_page(img, 300, (288, 360), "jpeg200", None, 5).info["jpeg_ppi"] == 200.0
    # the mask schemes ignore --dpi at every level, as before
    a = encode_page(img, 300, (288, 360), "mrc", 100, 7)
    b = encode_page(img, 300, (288, 360), "mrc", None, 7)
    assert [v.data for v in a.xobjects.values()] == [v.data for v in b.xobjects.values()]


def test_never_upsamples_low_resolution_pages():
    img = cv2.resize(colour_page(), (600, 750), interpolation=cv2.INTER_AREA)        # a 150 ppi scan
    e = encode_page(img, 150, (288, 360), "auto", None, 10)
    assert e.xobjects["Im0"].width == 600


# ----------------------------------------------------------------------------- end to end through the CLI
def make_input(tmp_path):
    pages = [colour_page(), synth_page(1200, 1500, colors=((0, 0, 0),), seed=1)]
    return make_scan_pdf(tmp_path / "in.pdf", pages, dpi=300)


def strip_id(b: bytes) -> bytes:
    return re.sub(rb"/ID \[<[0-9a-f]+><[0-9a-f]+>\]", b"/ID []", b)      # the trailer /ID differs between any two runs


@needs_jbig2
def test_cli_levels_valid_monotone_and_level_5_is_default(tmp_path):
    src = make_input(tmp_path)
    outs = {}
    for tag, extra in (("default", ""), ("q1", "--quality 1"), ("q5", "--quality 5"), ("q10", "--quality 10")):
        (tmp_path / tag).mkdir()
        w = tmp_path / tag / "scan.pdf"                              # same file name: the PDF title is the stem
        shutil.copy(src, w)
        r = run_cli(w, f"--jobs 2 --foreground --no-backup --keep-blank {extra}")
        assert r.returncode == 0, r.stderr
        with pikepdf.open(w) as pdf:
            assert len(pdf.pages) == 2 and pdf.check_pdf_syntax() == []
        assert image_filters(w) <= ALLOWED_FILTERS
        outs[tag] = w
    assert strip_id(outs["default"].read_bytes()) == strip_id(outs["q5"].read_bytes())
    assert outs["q1"].stat().st_size < outs["q5"].stat().st_size < outs["q10"].stat().st_size
    assert "/JBIG2Decode" in image_filters(outs["q5"]) and "/JBIG2Decode" not in image_filters(outs["q10"])


@needs_jbig2
def test_cli_report_records_quality_and_output_is_not_reprocessed(tmp_path):
    src = make_input(tmp_path)
    for q in (1, 10):
        w = tmp_path / f"q{q}.pdf"
        shutil.copy(src, w)
        rep = tmp_path / f"rep{q}.json"
        r = run_cli(w, f"--jobs 2 --foreground --no-backup --keep-blank --quality {q} --report {rep}")
        assert r.returncode == 0, r.stderr
        import json
        data = json.loads(rep.read_text())
        assert data["options"]["quality"] == q
        with pikepdf.open(w) as pdf:
            assert pdfio.scanpp_marker(pdf) is not None                     # marked, whatever the encoding
            assert pdfio.classify_pdf(pdf)[0] is False                      # ... so it is not taken for a plain scan again
        before = w.read_bytes()
        r = run_cli(w, f"--jobs 2 --foreground --no-backup --keep-blank --quality {q}")
        assert r.returncode == 0
        assert w.read_bytes() == before                                     # second run leaves plain-JPEG (q10) output untouched
