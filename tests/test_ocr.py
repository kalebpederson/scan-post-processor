"""--ocr: invisible searchable text layer (tesseract text-only PDF overlay)."""
import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pytest

from helpers import make_scan_pdf
from scanpp import cli, ocr, pdfio
from scanpp.pipeline import Options, process_file

needs_tess = pytest.mark.skipif(ocr.tesseract_languages() is None or "eng" not in (ocr.tesseract_languages() or ()),
                                reason="tesseract/eng missing")
needs_poppler = pytest.mark.skipif(shutil.which("pdftotext") is None or shutil.which("pdftoppm") is None,
                                   reason="poppler missing")
pytestmark = [needs_tess, needs_poppler]

DPI = 150
PDFTOTEXT = shutil.which("pdftotext")   # resolved up front: some tests empty PATH
PHRASES = [("Parallelogram", 150, 300), ("Opposite Sides", 150, 520), ("Reflexive Property", 150, 740)]


def phrase_page(w=1275, h=1650, phrases=PHRASES):
    img = np.full((h, w, 3), 255, np.uint8)
    for text, x, y in phrases:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def words(pdf, page=1):
    x = subprocess.run(["pdftotext", "-bbox", "-f", str(page), "-l", str(page), str(pdf), "-"],
                       capture_output=True, text=True).stdout
    return [(m[5], float(m[1]), float(m[2]), float(m[3]), float(m[4])) for m in
            re.finditer(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', x)]


def render(pdf, page=1, tool="pdftoppm", dpi=72):
    out = Path(pdf).with_suffix(f".{tool}.png")
    if tool == "pdftoppm":
        subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", "-singlefile", "-gray",
                        str(pdf), str(out.with_suffix(""))], check=True, capture_output=True)
    else:
        subprocess.run(["mutool", "draw", "-r", str(dpi), "-c", "gray", "-o", str(out), str(pdf), str(page)],
                       check=True, capture_output=True)
    return cv2.imread(str(out), 0).astype(int)


def overlay(src, dst, gray_dpi=DPI):
    """Overlay a tesseract text layer on page 1 of `src` (a one-image scan PDF) -> dst."""
    with pikepdf.open(src) as pdf:
        page = pdf.pages[0]
        img = pdfio.extract_page_input(page, 0)
        rgb = cv2.imdecode(np.frombuffer(img.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        layer, stats = ocr.ocr_page(rgb, img.dpi)
        assert layer is not None, stats
        form, ops = ocr.embed_layer(pdf, layer, img.media_box, img.rotate, {})
        page.Resources.XObject["/" + ocr.FORM_NAME] = form
        page.contents_add(pikepdf.Stream(pdf, ops), prepend=False)
        pdf.save(str(dst))
    return stats


def test_overlay_searchable_aligned_and_invisible(tmp_path):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page()], dpi=DPI)
    dst = tmp_path / "b.pdf"
    stats = overlay(src, dst)
    assert stats["words"] >= 3
    txt = subprocess.run(["pdftotext", str(dst), "-"], capture_output=True, text=True).stdout
    assert "Parallelogram" in txt and "Reflexive Property" in txt
    ws = {w[0]: w for w in words(dst)}
    s = 72.0 / DPI
    for word, x, y in [("Parallelogram", 150, 300), ("Reflexive", 150, 740)]:
        _, x0, y0, x1, y1 = ws[word]
        assert abs(x0 - x * s) < 4.0                                   # left edge (glyph side bearing allowed)
        assert y0 - 2 < (y - 20) * s < y1 + 2                          # box spans the glyph's vertical middle
        assert abs(y1 - y * s) < 4.0                                   # baseline
    for tool in ("pdftoppm", "mutool"):
        if shutil.which("mutool") or tool == "pdftoppm":
            assert np.abs(render(src, tool=tool) - render(dst, tool=tool)).max() <= 1


@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_rotated_pages(tmp_path, rotate):
    up = phrase_page()
    k = {90: 1, 180: 2, 270: 3}[rotate]                     # stored image = upright rotated back (CCW k quarter turns)
    stored = np.ascontiguousarray(np.rot90(up, k))
    src = make_scan_pdf(tmp_path / "a.pdf", [stored], dpi=DPI, rotate=rotate)
    out = tmp_path / "o.pdf"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--no-deskew", "--ocr", "--no-notify"]) == 0
    with pikepdf.open(out) as pdf:
        assert int(pdf.pages[0].Rotate) == rotate
    ws = {w[0]: w for w in words(out)}
    assert "Parallelogram" in ws, ws
    _, x0, y0, x1, y1 = ws["Parallelogram"]
    s = 72.0 / DPI                                          # upright display geometry
    assert abs(x0 - 150 * s) < 4.0 and abs(y1 - 300 * s) < 4.0
    # invisible: identical render with and without OCR
    plain = tmp_path / "p.pdf"
    assert cli.main([str(src), "-o", str(plain), "--jobs", "1", "--no-deskew", "--no-notify"]) == 0
    assert np.abs(render(out) - render(plain)).max() <= 1


def test_non_letter_mediabox(tmp_path):
    img = phrase_page(w=1240, h=1754)                       # A4 @150 dpi
    src = make_scan_pdf(tmp_path / "a.pdf", [img], dpi=DPI)
    out = tmp_path / "o.pdf"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--ocr", "--no-notify"]) == 0
    with pikepdf.open(out) as pdf:
        mb = [float(v) for v in pdf.pages[0].MediaBox]
    assert abs(mb[2] - 595.2) < 1 and abs(mb[3] - 842.4) < 1
    ws = {w[0]: w for w in words(out)}
    assert abs(ws["Parallelogram"][1] - 150 * 72 / DPI) < 4.0


def test_nonzero_mediabox_origin(tmp_path):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page()], dpi=DPI)
    with pikepdf.open(src, allow_overwriting_input=True) as pdf:        # shift the box, keep size
        p = pdf.pages[0]
        w, h = float(p.MediaBox[2]), float(p.MediaBox[3])
        p.MediaBox = [100, 50, 100 + w, 50 + h]
        p.Contents = pdf.make_stream(b"q 1 0 0 1 100 50 cm\n" + p.Contents.read_bytes() + b"\nQ")
        pdf.save(str(tmp_path / "s.pdf"))
    out = tmp_path / "o.pdf"
    assert cli.main([str(tmp_path / "s.pdf"), "-o", str(out), "--jobs", "1", "--no-deskew", "--ocr", "--no-notify"]) == 0
    with pikepdf.open(out) as pdf:
        assert [float(v) for v in pdf.pages[0].MediaBox][:2] == [100, 50]
    ws = {w[0]: w for w in words(out)}
    assert abs(ws["Parallelogram"][1] - 150 * 72 / DPI) < 4.0      # pdftotext reports relative to the box origin


def test_page_without_text_gets_no_layer(tmp_path):
    img = np.full((1650, 1275, 3), 255, np.uint8)
    cv2.circle(img, (600, 800), 300, (0, 0, 0), 12)
    cv2.rectangle(img, (200, 200), (500, 400), (0, 0, 0), 10)
    src = make_scan_pdf(tmp_path / "a.pdf", [img], dpi=DPI)
    out = tmp_path / "o.pdf"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--ocr", "--no-notify"]) == 0
    with pikepdf.open(out) as pdf:
        assert not pdfio.has_scanpp_text_layer(pdf.pages[0])
        assert "/Font" not in pdf.pages[0].Resources


def test_blank_kept_page_not_ocrd(tmp_path):
    from helpers import blank_page
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page(), blank_page(w=1275, h=1650)], dpi=DPI)
    out = tmp_path / "o.pdf"
    rep = tmp_path / "r.json"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--ocr", "--keep-blank", "--no-notify",
                     "--report", str(rep)]) == 0
    import json
    pages = json.loads(rep.read_text())["files"][0]["pages"]
    assert "ocr" in pages[0] and "ocr" not in pages[1]


def test_report_has_ocr_time_and_words(tmp_path):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page()], dpi=DPI)
    rep = tmp_path / "r.json"
    assert cli.main([str(src), "-o", str(tmp_path / "o.pdf"), "--jobs", "1", "--ocr", "--no-notify",
                     "--report", str(rep)]) == 0
    import json
    f = json.loads(rep.read_text())["files"][0]
    o = f["pages"][0]["ocr"]
    assert o["words"] >= 3 and o["cpu"] > 0 and o["wall"] > 0 and "conf_mean" in o
    assert f["stage_cpu"]["ocr"] > 0


def test_font_shared_between_pages(tmp_path):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page(), phrase_page()], dpi=DPI)
    out = tmp_path / "o.pdf"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--ocr", "--no-notify"]) == 0
    with pikepdf.open(out) as pdf:
        fonts = [next(iter(p.Resources.XObject[f"/{ocr.FORM_NAME}"].Resources.Font.values())) for p in pdf.pages]
        assert fonts[0].objgen == fonts[1].objgen


# ----------------------------------------------------------------------------- classifier
def test_ocrd_output_is_not_reprocessed(tmp_path):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page()], dpi=DPI)
    out = tmp_path / "o.pdf"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--ocr", "--no-notify"]) == 0
    before = out.read_bytes()
    with pikepdf.open(out) as pdf:
        ok, why = pdfio.classify_pdf(pdf)
    assert not ok and "scanpp" in why
    res = process_file(out, Options(jobs=1, ocr=True, backup=False))
    assert res.status == "skipped" and out.read_bytes() == before


def test_digital_pdf_with_text_still_skipped(tmp_path):
    from helpers import make_digital_pdf
    make_digital_pdf(tmp_path / "d.pdf")
    with pikepdf.open(tmp_path / "d.pdf") as pdf:
        ok, why = pdfio.classify_pdf(pdf)
    assert not ok


# ----------------------------------------------------------------------------- failure paths
def _run_ocr(tmp_path, *extra):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page(), phrase_page()], dpi=DPI)
    out = tmp_path / "o.pdf"
    rc = cli.main([str(src), "-o", str(out), "--jobs", "1", "--ocr", "--no-notify", *extra])
    return rc, out


def _no_text(out):
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 2
        assert not any(pdfio.has_scanpp_text_layer(p) for p in pdf.pages)
    return subprocess.run([PDFTOTEXT, str(out), "-"], capture_output=True, text=True).stdout.strip() == ""


def test_missing_language_warns_once_and_continues(tmp_path, capsys):
    rc, out = _run_ocr(tmp_path, "--ocr-lang", "xyz")
    assert rc == 0 and _no_text(out)
    err = capsys.readouterr().err
    assert err.count("language data missing") == 1


def test_tesseract_absent_warns_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    (tmp_path / "empty").mkdir()
    rc, out = _run_ocr(tmp_path)
    assert rc == 0 and _no_text(out)
    assert capsys.readouterr().err.count("tesseract is not installed") == 1


def test_tesseract_crash_per_page_is_not_fatal(tmp_path, monkeypatch, capsys):
    real = shutil.which("tesseract")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "tesseract"
    fake.write_text(f'#!/bin/sh\nif [ "$1" = "--list-langs" ]; then exec {real} --list-langs; fi\n'
                    'echo boom >&2; exit 3\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    rc, out = _run_ocr(tmp_path)
    assert rc == 0 and _no_text(out)
    assert capsys.readouterr().err.count("OCR failed for a page") == 2


def test_ocr_off_by_default(tmp_path):
    src = make_scan_pdf(tmp_path / "a.pdf", [phrase_page()], dpi=DPI)
    out = tmp_path / "o.pdf"
    assert cli.main([str(src), "-o", str(out), "--jobs", "1", "--no-notify"]) == 0
    with pikepdf.open(out) as pdf:
        assert not pdfio.has_scanpp_text_layer(pdf.pages[0])


def test_conf_filter_drops_low_confidence_words():
    img = phrase_page()
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 0, img.shape), 0, 255).astype(np.uint8)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    all_, _ = ocr.ocr_page(g, DPI, min_conf=0)
    none, st = ocr.ocr_page(g, DPI, min_conf=101)
    assert all_ is not None and none is None and st["words_dropped_low_conf"] == st["words_raw"] > 0
