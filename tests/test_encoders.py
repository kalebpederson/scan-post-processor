"""Page encoders: mask threshold, JBIG2/G4 codecs, per-page selection, PDF validity, rendering fidelity, colour identity."""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pytest

from scanpp import encode, mrc, pdfio
from scanpp.encode import encode_page

needs_render = pytest.mark.skipif(shutil.which("pdftoppm") is None or shutil.which("mutool") is None,
                                  reason="pdftoppm/mutool missing")
needs_jbig2 = pytest.mark.skipif(mrc.jbig2_path() is None, reason="jbig2 missing")


# ----------------------------------------------------------------------------- helpers
def write_pages(path, pages, dpi=300.0, encoder="auto", rotate=0):
    """Encode full-res cleaned pages and write a PDF. Returns the list of EncodedPages."""
    eps = []
    for rgb in pages:
        h, w = rgb.shape[:2]
        ep = encode_page(rgb, dpi, (w * 72 / dpi, h * 72 / dpi), encoder)
        ep.media_box = (0.0, 0.0, w * 72 / dpi, h * 72 / dpi)
        ep.rotate = rotate
        eps.append(ep)
    pdfio.write_pdf(eps, path, {"title": "t"})
    return eps


def render(pdf, renderer, dpi, page=1):
    base = Path(pdf).with_suffix("")
    if renderer == "pdftoppm":
        out = f"{base}_p{page}"
        subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png", "-singlefile",
                        str(pdf), out], check=True, capture_output=True)
        out += ".png"
    else:
        out = f"{base}_m{page}.png"
        subprocess.run(["mutool", "draw", "-r", str(dpi), "-o", out, str(pdf), str(page)], check=True,
                       capture_output=True)
    img = cv2.imread(out, cv2.IMREAD_COLOR)
    assert img is not None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def check_syntax(path):
    with pikepdf.open(path) as pdf:
        assert pdf.check_pdf_syntax() == []


def synth_page(w=1200, h=1500, colors=((0, 0, 0),), highlighter=False, seed=0):
    """White page, 300 ppi-like strokes: text lines in `colors` (cycled) plus slightly anti-aliased pencil."""
    img = np.full((h, w, 3), 255, np.uint8)
    y = 100
    i = 0
    while y < h - 100:
        cv2.putText(img, "Algebra x=4 triangle ABC", (80, y), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    colors[i % len(colors)], 3, cv2.LINE_AA)
        y += 90
        i += 1
    if highlighter:
        cv2.rectangle(img, (100, 300), (700, 380), (255, 250, 120), -1)
        cv2.putText(img, "Algebra x=4 triangle ABC", (110, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def hue_deg(rgb):
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    return float(hsv[0]) * 2.0


def hue_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


# ----------------------------------------------------------------------------- unit: mask threshold
def test_mask_threshold_on_synthetic_strokes():
    img = np.full((120, 200, 3), 255, np.uint8)
    img[20:24, 10:100] = 0                      # black stroke (d = 255)
    img[19, 10:100] = 255 - 80                  # edge pixel next to it: d = 80  < 88 -> not ink
    img[25, 10:100] = 255 - 95                  # d = 95 >= 88 -> ink
    img[60:63, 10:100] = 255 - 40               # lone pencil stroke: threshold clip(.35*40,20,88)=20 -> ink
    img[80:83, 10:100] = 255 - 15               # d = 15 < 20 -> dropped
    img[100:103, 10:100] = 255 - 130            # mid gray stroke: thr = 45.5 -> ink
    ink = mrc.ink_mask(img, 300.0)
    assert ink[20:24, 10:100].all()
    assert not ink[19, 10:100].any() and ink[25, 10:100].all()
    assert ink[60:63, 10:100].all()
    assert not ink[80:83, 10:100].any()
    assert ink[100:103, 10:100].all()
    assert not ink[0:10].any() and not ink[:, 150:].any()


def test_mask_threshold_window_scales_with_ppi():
    assert mrc._odd_window(7, 300) == 7 and mrc._odd_window(7, 150) == 3 and mrc._odd_window(7, 200) == 5
    assert mrc._odd_window(7, 75) == 3


# ----------------------------------------------------------------------------- codecs
@needs_render
@needs_jbig2
def test_g4_and_jbig2_masks_are_bit_exact(tmp_path):
    rng = np.random.default_rng(3)
    ink = np.zeros((700, 530), bool)            # width not a multiple of 8
    for _ in range(60):
        x, y = rng.integers(0, 500), rng.integers(0, 680)
        cv2.circle(ink.view(np.uint8), (int(x), int(y)), int(rng.integers(1, 12)), 1, int(rng.integers(-1, 3)))
    ink[0, :] = True
    ink[:, -1] = True                           # edges
    h, w = ink.shape
    ref = np.where(ink, 0, 255).astype(np.uint8)
    outs = {}
    for name, (data, codec, kw) in {"j": (mrc.jbig2_generic(ink), "jbig2", dict(filter="JBIG2Decode", parms=None)),
                                    "g": (mrc.g4_strip(ink), "g4", None)}.items():
        extra = {"ImageMask": True}
        if name == "g":
            kw = dict(filter="CCITTFaxDecode", parms={"K": -1, "Columns": w, "Rows": h, "BlackIs1": False})
        if kw["parms"]:
            extra["DecodeParms"] = kw["parms"]
        pdf = pikepdf.Pdf.new()
        st = pdfio._make_image(pdf, encode.ImageSpec(data, w, h, None, 1, kw["filter"], extra))
        page = pikepdf.Dictionary(
            Type=pikepdf.Name.Page, MediaBox=[0, 0, w, h],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=st)),
            Contents=pikepdf.Stream(pdf, f"0 g q {w} 0 0 {h} 0 0 cm /Im0 Do Q".encode()))
        pdf.pages.append(pikepdf.Page(page))
        f = tmp_path / f"{name}.pdf"
        pdf.save(f)
        check_syntax(f)
        for r in ("pdftoppm", "mutool"):
            img = render(f, r, 72)[..., 0]
            assert img.shape == ref.shape
            assert (img == ref).all(), (name, r, int((img != ref).sum()))
            outs[(name, r)] = img
    assert (outs[("j", "pdftoppm")] == outs[("g", "mutool")]).all()


def test_jbig2_is_generic_embedded_stream():
    ink = np.zeros((64, 64), bool)
    ink[10:20, 5:50] = True
    data = mrc.jbig2_generic(ink)
    assert data[:4] == b"\x00\x00\x00\x00"       # segment number 0
    assert data[4] in (0x30, 0x31, 0x32, 0x33, 0x00, 0x36, 0x38, 0x39, 0x3a, 0x3b, 0x28, 0x2a)  # a region/page segment
    assert b"%PDF" not in data


@needs_render
def test_g4_fallback_when_jbig2_missing(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(mrc, "jbig2_path", lambda: None)
    mrc._warned.discard("g4")
    img = synth_page()
    with caplog.at_level(logging.WARNING, logger="scanpp"):
        eps = write_pages(tmp_path / "g4.pdf", [img, img], encoder="mrc")
    assert [e.info["mask_codec"] for e in eps] == ["g4", "g4"]
    assert sum("falling back to CCITT G4" in r.message for r in caplog.records) == 1       # logged once
    with pikepdf.open(tmp_path / "g4.pdf") as pdf:
        m = pdf.pages[0].Resources.XObject.Fg.Mask
        assert m.Filter == "/CCITTFaxDecode" and m.DecodeParms.K == -1 and m.DecodeParms.Columns == 1200
        assert m.DecodeParms.Rows == 1500 and not m.DecodeParms.BlackIs1
    check_syntax(tmp_path / "g4.pdf")
    monkeypatch.undo()
    if mrc.jbig2_path() is not None:                                   # the real jbig2 exists: renders must be identical
        write_pages(tmp_path / "j.pdf", [img], encoder="mrc")
        for r in ("pdftoppm", "mutool"):
            assert (render(tmp_path / "j.pdf", r, 300) == render(tmp_path / "g4.pdf", r, 300)).all()


def test_g4_used_when_jbig2_command_fails(monkeypatch):
    class R:
        returncode, stdout, stderr = 1, b"", b"boom"
    monkeypatch.setattr(mrc.subprocess, "run", lambda *a, **k: R)
    monkeypatch.setattr(mrc, "jbig2_path", lambda: "/bin/false")
    ink = np.zeros((40, 40), bool)
    ink[5:9, 5:30] = True
    data, codec, _ = mrc.encode_mask(ink)
    assert codec == "g4" and len(data) > 0


# ----------------------------------------------------------------------------- selection
def test_selection_rules():
    black = synth_page()
    assert encode.select_scheme(black, 300) == "e3"                                    # gray-only page -> gray mrc
    colored = synth_page(colors=((0, 0, 0), (200, 20, 40), (20, 90, 200)))
    assert encode.select_scheme(colored, 300) == "e3x"                                 # coloured pen -> mrc
    hl = synth_page(colors=((0, 0, 0), (200, 20, 40)), highlighter=True)
    assert encode.select_scheme(hl, 300) == "e1"                                       # highlighter -> jpeg
    assert encode.select_scheme(np.full((300, 300, 3), 255, np.uint8), 300) == "blank"
    heavy = colored.copy()
    heavy[:, :600] = (30, 30, 200)                                                     # ink-heavy art page
    assert encode.select_scheme(heavy, 300) == "e1"


def test_highlighter_thresholds_scale_with_dpi():
    img = np.full((750, 600, 3), 255, np.uint8)            # a 150 ppi page: 300x40 px highlighter = 12000 px @300 ppi
    cv2.rectangle(img, (50, 100), (350, 140), (255, 250, 120), -1)
    assert encode.has_highlighter(img, 150.0)
    cv2.rectangle(img, (50, 300), (250, 306), (255, 250, 120), -1)          # a thin line is not a highlighter
    img2 = np.full((750, 600, 3), 255, np.uint8)
    cv2.rectangle(img2, (50, 300), (250, 306), (255, 250, 120), -1)
    assert not encode.has_highlighter(img2, 150.0)


def test_encoder_names_and_default():
    assert {"auto", "mrc", "jpeg", "jpeg200"} <= set(encode.encoder_names())
    assert encode.DEFAULT_ENCODER == "auto"


def test_scheme_chosen_by_encoder():
    img = synth_page(colors=((0, 0, 0), (200, 20, 40)))
    assert encode_page(img, 300, (288, 360), "auto").info["scheme"] == "e3x"
    assert encode_page(img, 300, (288, 360), "mrc").info["scheme"] == "e3x"
    j = encode_page(img, 300, (288, 360), "jpeg")
    assert j.info["scheme"] == "e1" and j.info["jpeg_ppi"] == 150 and j.xobjects["Im0"].width == 600
    j2 = encode_page(img, 300, (288, 360), "jpeg200")
    assert j2.xobjects["Im0"].width == 800
    assert encode_page(img, 300, (288, 360), "jpeg", jpeg_dpi=100).xobjects["Im0"].width == 400
    assert encode_page(synth_page(), 300, (288, 360), "mrc").info["scheme"] == "e3"
    hl = synth_page(colors=((0, 0, 0), (200, 20, 40)), highlighter=True)
    assert encode_page(hl, 300, (288, 360), "mrc").info["scheme"] == "e3x"             # mrc ignores the highlighter rule
    assert encode_page(hl, 300, (288, 360), "auto").info["scheme"] == "e1"


def test_jpeg_is_444_baseline():
    img = synth_page(colors=((0, 0, 0), (200, 20, 40)))
    data = encode_page(img, 300, (288, 360), "jpeg").xobjects["Im0"].data
    # SOF0 (baseline) present, no SOF2 (progressive); 3 components all with sampling 0x11 (4:4:4)
    i = data.index(b"\xff\xc0")
    assert b"\xff\xc2" not in data[:i + 2]
    comps = data[i + 10: i + 19]
    assert [comps[k] for k in (1, 4, 7)] == [0x11, 0x11, 0x11]


def test_mrc_layers_are_4_4_4_baseline_and_100ppi():
    img = synth_page(colors=((0, 0, 0), (200, 20, 40)))
    ep = encode_page(img, 300, (288, 360), "mrc")
    fg = ep.xobjects["Fg"]
    assert (fg.width, fg.height) == (400, 500) and fg.mask.width == 1200 and fg.mask.height == 1500
    data = fg.data
    i = data.index(b"\xff\xc0")
    assert [data[i + 10 + k] for k in (1, 4, 7)] == [0x11, 0x11, 0x11]


@pytest.mark.parametrize("native,mask_w,col_w", [(300, 1200, 400), (200, 800, 400), (150, 600, 400), (100, 400, 400)])
def test_no_upsampling_below_300dpi(native, mask_w, col_w):
    full = synth_page(w=int(4 * native), h=int(5 * native), colors=((0, 0, 0), (200, 20, 40)))
    ep = encode_page(full, native, (288, 360), "mrc")
    fg = ep.xobjects["Fg"]
    assert fg.mask.width == int(4 * native)                      # mask at native resolution
    assert fg.width == round(4 * native * min(100, native) / native)   # colour layer at min(100, native) ppi
    assert ep.info["mask_ppi"] == native


def test_oversampled_input_mask_capped_at_300():
    full = synth_page(w=2400, h=3000, colors=((0, 0, 0), (200, 20, 40)))
    ep = encode_page(full, 600, (288, 360), "mrc")
    assert ep.xobjects["Fg"].mask.width == 1200


def test_blank_page_has_empty_content(tmp_path):
    ep = encode_page(np.full((300, 200, 3), 255, np.uint8), 100, (144, 216), "auto")
    assert ep.xobjects == {} and ep.content == b"" and ep.info["scheme"] == "blank"
    for enc in ("mrc", "jpeg", "jpeg200"):
        assert encode_page(np.full((300, 200, 3), 255, np.uint8), 100, (144, 216), enc).info["scheme"] == "blank"


# ----------------------------------------------------------------------------- PDF structure / validity
@needs_render
def test_pdf_structure_and_validity(tmp_path):
    pages = [synth_page(), synth_page(colors=((0, 0, 0), (200, 20, 40))),
             synth_page(colors=((0, 0, 0), (200, 20, 40)), highlighter=True), np.full((1500, 1200, 3), 255, np.uint8)]
    eps = write_pages(tmp_path / "s.pdf", pages)
    assert [e.info["scheme"] for e in eps] == ["e3", "e3x", "e1", "blank"]
    f = tmp_path / "s.pdf"
    check_syntax(f)
    pdfio.validate_pdf(f, 4)
    with pikepdf.open(f) as pdf:
        p = pdf.pages[1]
        fg = p.Resources.XObject.Fg
        assert fg.Subtype == "/Image" and fg.ColorSpace == "/DeviceRGB" and fg.BitsPerComponent == 8
        assert fg.Filter == "/DCTDecode" and (int(fg.Width), int(fg.Height)) == (400, 500)
        m = fg.Mask
        assert m.ImageMask and m.BitsPerComponent == 1 and (int(m.Width), int(m.Height)) == (1200, 1500)
        assert m.Filter == ("/JBIG2Decode" if mrc.jbig2_path() else "/CCITTFaxDecode")
        assert "/Decode" not in m and "/ColorSpace" not in m
        assert [float(v) for v in p.MediaBox] == pytest.approx([0, 0, 288, 360], abs=1e-3)
        assert b"/Fg Do" in p.Contents.read_bytes()
        assert pdf.pages[0].Resources.XObject.Fg.ColorSpace == "/DeviceGray"
        assert pdf.pages[2].Resources.XObject.Im0.Filter == "/DCTDecode"
        assert pdf.pages[3].Contents.read_bytes() == b"" and len(pdf.pages[3].Resources.XObject) == 0
        assert not pdfio.classify_pdf(pdf)[0]            # our own output is never re-processed as a plain scan


# ----------------------------------------------------------------------------- rendering fidelity
def _quality(ref, got):
    r = ref.min(axis=2)
    g = got.min(axis=2)
    solid = r < 128
    ink_loss = float((g[solid] >= 245).mean())
    speckle = float((g[r == 255] < 245).mean())
    return ink_loss, speckle, float(np.abs(ref.astype(np.int16) - got).mean())


@needs_render
@pytest.mark.parametrize("dpi", [300, 200, 150])
@pytest.mark.parametrize("renderer", ["pdftoppm", "mutool"])
def test_mrc_render_matches_cleaned_reference(tmp_path, renderer, dpi):
    s = dpi / 300
    ref = synth_page(w=int(1200 * s), h=int(1500 * s), colors=((0, 0, 0), (200, 20, 40), (20, 120, 40)))
    ref = cv2.GaussianBlur(ref, (0, 0), 0.6)               # soft edges like a cleaned scan
    eps = write_pages(tmp_path / "q.pdf", [ref], dpi=dpi, encoder="auto")
    assert eps[0].info["scheme"] == "e3x"
    got = render(tmp_path / "q.pdf", renderer, dpi)
    assert got.shape == ref.shape
    ink_loss, speckle, mad = _quality(ref, got)
    assert ink_loss < 0.01 and speckle < 0.005, (ink_loss, speckle)
    # mask edge fidelity: luminance difference only on pixels near ink
    near = cv2.dilate((ref.min(axis=2) < 200).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge_mad = np.abs(ref.min(axis=2).astype(np.int16) - got.min(axis=2))[near].mean()
    assert edge_mad < 35, edge_mad
    assert mad < 6, mad


@needs_render
@pytest.mark.parametrize("renderer", ["pdftoppm", "mutool"])
def test_gray_mrc_and_jpeg_render(tmp_path, renderer):
    ref = cv2.GaussianBlur(synth_page(), (0, 0), 0.6)
    for enc in ("auto", "jpeg", "jpeg200"):
        write_pages(tmp_path / f"{enc}.pdf", [ref], encoder=enc)
        check_syntax(tmp_path / f"{enc}.pdf")
        got = render(tmp_path / f"{enc}.pdf", renderer, 300)
        ink_loss, speckle, mad = _quality(ref, got)
        assert ink_loss < 0.02 and speckle < 0.02 and mad < 8, (enc, ink_loss, speckle, mad)


@needs_render
@pytest.mark.parametrize("renderer", ["pdftoppm", "mutool"])
@pytest.mark.parametrize("encoder", ["auto", "jpeg"])
@pytest.mark.parametrize("rotate", [90, 180, 270])
def test_rotated_pages_render_upright(tmp_path, renderer, encoder, rotate):
    """Image stored sideways + /Rotate: the displayed page must equal the upright reference."""
    upright = cv2.GaussianBlur(synth_page(colors=((0, 0, 0), (200, 20, 40))), (0, 0), 0.6)
    k = {90: 1, 180: 2, 270: 3}[rotate]
    stored = np.ascontiguousarray(np.rot90(upright, k))     # CCW k*90, so a CW /Rotate shows it upright
    write_pages(tmp_path / "r.pdf", [stored], encoder=encoder, rotate=rotate)
    with pikepdf.open(tmp_path / "r.pdf") as pdf:
        assert int(pdf.pages[0].Rotate) == rotate
    check_syntax(tmp_path / "r.pdf")
    got = render(tmp_path / "r.pdf", renderer, 300)
    assert got.shape == upright.shape
    ink_loss, speckle, mad = _quality(upright, got)
    assert ink_loss < 0.03 and speckle < (0.005 if encoder == "auto" else 0.05) and mad < 10, (ink_loss, speckle, mad)


# ----------------------------------------------------------------------------- colour identity regression
@needs_render
@pytest.mark.parametrize("renderer", ["pdftoppm", "mutool"])
@pytest.mark.parametrize("encoder", ["auto", "mrc", "jpeg", "jpeg200"])
def test_adjacent_strokes_keep_their_hue(tmp_path, renderer, encoder):
    """Adjacent magenta / red / green / blue pen strokes: 4:2:0 or a mean-over-blobs colour layer swaps or
    desaturates them.  After encode+render each stroke's median hue must stay within 20 deg of the original."""
    cols = {"magenta": (220, 30, 170), "red": (225, 30, 30), "green": (30, 150, 50), "blue": (30, 60, 220)}
    img = np.full((600, 800, 3), 255, np.uint8)
    boxes = {}
    x = 100
    for name, c in cols.items():
        cv2.line(img, (x, 80), (x + 10, 520), c, 5, cv2.LINE_AA)          # 5 px strokes, 3 px apart
        boxes[name] = (x, c)
        x += 8
    cv2.putText(img, "x=2", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)
    write_pages(tmp_path / "c.pdf", [img], encoder=encoder)
    got = render(tmp_path / "c.pdf", renderer, 300)
    for name, (x0, c) in boxes.items():
        ys = np.arange(120, 480)
        xs = (x0 + 10 * (ys - 80) / 440).round().astype(int)           # stroke centre line
        px = got[ys, xs]
        med = np.median(px, axis=0)
        ref_h = hue_deg(c)
        assert hue_diff(hue_deg(tuple(int(v) for v in med)), ref_h) <= 20, (name, med, c)
        assert med.max() - med.min() > 0.6 * (max(c) - min(c)), (name, "desaturated", med)


@needs_jbig2
@pytest.mark.parametrize("first", [0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20])
def test_jbig2_mask_whose_first_raster_byte_is_whitespace(first):
    """leptonica's PNM reader mistakes such a first raster byte for header whitespace (-> G4 fallback); jbig2 gets a PNG then."""
    ink = np.zeros((16, 64), bool)
    ink[:, :8] = np.unpackbits(np.array([first], np.uint8)).astype(bool)
    assert mrc.jbig2_generic(ink)
