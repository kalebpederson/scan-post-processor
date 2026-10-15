"""Robustness: tiny images, image modes (alpha/palette/16-bit), scanpp markers (no re-processing), paper boundary."""
import hashlib
import io

import cv2
import numpy as np
import pikepdf
import pytest
from PIL import Image

from helpers import blank_page, make_scan_pdf, text_page
from scanpp import boundary, geometry, imageio, ocr, pdfio, pipeline
from scanpp.options import Options


# ----------------------------------------------------------------------------- tiny images: nothing may raise
SIZES = [16, 17, 33, 64, 77, 81, 100, 150, 231, 320, 400]


@pytest.mark.parametrize("dpi", [72, 100, 150, 300, 600])
def test_geometry_functions_safe_on_tiny_images(dpi):
    rng = np.random.default_rng(0)
    for w in SIZES:
        for h in (16, 81, 231, 400):
            for kind in ("noise", "white", "text"):
                if kind == "noise":
                    im = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
                elif kind == "white":
                    im = np.full((h, w, 3), 250, np.uint8)
                else:
                    im = np.ascontiguousarray(text_page(w=max(w, 16), h=max(h, 16))[:h, :w])
                g, d = geometry.prep_gray(im, dpi)
                geometry.estimate_skew(g, d)
                geometry.blank_score(im, dpi)
                mask, _ = geometry.edge_artifact_mask(im, dpi)
                assert mask.shape == (h, w)
                out, info = geometry.clean_edges(im, dpi)
                assert out.shape == im.shape
                geometry.deskew(im, 0.5)
                geometry.paper_colour(im)


@pytest.mark.parametrize("dpi", [72, 150, 300])
def test_process_array_safe_on_tiny_images(dpi):
    rng = np.random.default_rng(1)
    for w, h in [(16, 16), (16, 300), (300, 16), (40, 50), (77, 81), (100, 100), (228, 231), (320, 200), (400, 400), (399, 17)]:
        for im in (rng.integers(0, 255, (h, w, 3), dtype=np.uint8), np.full((h, w, 3), 255, np.uint8),
                   np.ascontiguousarray(text_page(w=max(w, 16), h=max(h, 16))[:h, :w])):
            out, odpi, stats = pipeline.process_array(im.copy(), dpi, Options(), allow_drop=False)
            assert out.shape[2] == 3 and out.dtype == np.uint8 and out.size > 0


def test_coil_extent_short_run_returns_zero():
    for n in (3, 10, 50, 77, 120):
        assert geometry._coil_extent(np.zeros(n, int), 100) == 0
        assert geometry._coil_extent(np.arange(n) % 5, 300) == 0


# ----------------------------------------------------------------------------- image modes
def _save(im, tmp_path, name, **kw):
    p = tmp_path / name
    im.save(p, **kw)
    return p


def test_rgba_alpha_composited_on_white(tmp_path):
    a = np.zeros((60, 60, 4), np.uint8)
    a[..., :3] = 0                                   # black under the transparent area (would show if alpha were dropped)
    a[20:40, 20:40] = (200, 30, 30, 255)
    p = _save(Image.fromarray(a, "RGBA"), tmp_path, "a.png", dpi=(150, 150))
    rgb, dpi, fmt = imageio.read_image(p)
    assert fmt == "PNG" and dpi == pytest.approx(150, abs=0.1)
    assert (rgb[0, 0] == 255).all() and tuple(rgb[30, 30]) == (200, 30, 30)
    a[..., 3] = 128
    rgb, _, _ = imageio.read_image(_save(Image.fromarray(a, "RGBA"), tmp_path, "b.png"))
    assert abs(int(rgb[0, 0, 0]) - 127) <= 2         # 50 % black over white


def test_la_and_palette_with_transparency(tmp_path):
    la = np.zeros((30, 30, 2), np.uint8)
    la[..., 0] = 0
    rgb, _, _ = imageio.read_image(_save(Image.fromarray(la, "LA"), tmp_path, "la.png"))
    assert (rgb == 255).all()                        # fully transparent -> white
    pal = Image.new("P", (30, 30), 0)
    pal.putpalette([0, 0, 0, 255, 0, 0] + [0] * 762)
    pal.paste(1, (10, 10, 20, 20))
    rgb, _, _ = imageio.read_image(_save(pal, tmp_path, "p.png", transparency=0))
    assert (rgb[0, 0] == 255).all() and tuple(rgb[15, 15]) == (255, 0, 0)
    rgb, _, _ = imageio.read_image(_save(pal, tmp_path, "p2.png"))      # opaque palette
    assert (rgb[0, 0] == 0).all() and tuple(rgb[15, 15]) == (255, 0, 0)


def test_16bit_gray_scaled_not_clipped(tmp_path):
    ramp = np.tile(np.linspace(0, 65535, 64).astype(np.uint16), (64, 1))
    p = tmp_path / "g16.png"
    Image.fromarray(ramp).save(p)
    rgb, _, _ = imageio.read_image(p)
    g = rgb[0, :, 0].astype(int)
    assert g[0] == 0 and g[-1] == 255 and (np.diff(g) >= 0).all() and 100 < g[32] < 160     # no blown highlights
    rgb2, _, _ = imageio.read_image(_save(Image.fromarray(ramp.astype(np.uint8)), tmp_path, "g8.png"))
    assert rgb2.shape[2] == 3


def test_cmyk_and_1bit_modes(tmp_path):
    im = Image.fromarray(np.full((20, 20, 3), (10, 200, 10), np.uint8)).convert("CMYK")
    rgb, _, _ = imageio.read_image(_save(im, tmp_path, "c.jpg"))
    assert rgb.shape == (20, 20, 3) and rgb[0, 0, 1] > 150
    bw = Image.fromarray(np.tile(np.array([0, 255], np.uint8), (10, 10))).convert("1")
    rgb, _, _ = imageio.read_image(_save(bw, tmp_path, "b.png"))
    assert set(np.unique(rgb)) == {0, 255}


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "I;16", "L", "RGB"])
def test_process_image_file_modes_keep_dpi(tmp_path, mode):
    base = text_page(w=420, h=300)
    if mode == "RGBA":
        im = Image.fromarray(np.dstack([base, np.full(base.shape[:2], 255, np.uint8)]), "RGBA")
    elif mode == "LA":
        im = Image.fromarray(np.dstack([cv2.cvtColor(base, cv2.COLOR_RGB2GRAY), np.full(base.shape[:2], 255, np.uint8)]), "LA")
    elif mode == "P":
        im = Image.fromarray(base).convert("P")
    elif mode == "I;16":
        im = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.uint16) * 257)
    else:
        im = Image.fromarray(base).convert(mode)
    p = _save(im, tmp_path, "x.png", dpi=(100, 100))
    res = pipeline.process_file(p, Options(backup=False, jobs=1, notify=False))
    assert res.status == "processed", res.reason
    with Image.open(p) as o:
        assert o.format == "PNG" and o.size == (420, 300)
        assert o.info["dpi"][0] == pytest.approx(100, abs=0.2)


def test_jpeg_dpi_preserved(tmp_path):
    p = _save(Image.fromarray(text_page(w=420, h=300)), tmp_path, "x.jpg", dpi=(150, 150))
    pipeline.process_file(p, Options(backup=False, jobs=1, notify=False))
    with Image.open(p) as o:
        assert o.format == "JPEG" and o.info["dpi"][0] == pytest.approx(150, abs=0.2)


# ----------------------------------------------------------------------------- scanpp marker: output is never re-processed
def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()




@pytest.mark.parametrize("encoder", ["auto", "mrc", "jpeg", "jpeg200"])
@pytest.mark.parametrize("with_ocr", [False, True])
@pytest.mark.parametrize("keep_blank", [False, True])
def test_second_run_skips_and_is_byte_identical(tmp_path, encoder, with_ocr, keep_blank):
    if with_ocr and ocr.check_available("eng"):
        pytest.skip("tesseract not available")
    pages = [text_page(seed=1), blank_page(), text_page(seed=2, paper=(240, 230, 200))]
    p = make_scan_pdf(tmp_path / "s.pdf", pages)
    opts = Options(encoder=encoder, ocr=with_ocr, drop_blank=not keep_blank, jobs=1, backup=False, notify=False)
    r1 = pipeline.process_file(p, opts)
    assert r1.status == "processed"
    assert r1.pages_out == (3 if keep_blank else 2)
    h1 = _sha(p)
    with pikepdf.open(p) as pdf:
        assert pdfio.scanpp_marker(pdf)
        assert "/ScanppVersion" in pdf.docinfo and all("/ScanppPage" in pg.obj for pg in pdf.pages)
    r2 = pipeline.process_file(p, opts)
    assert r2.status == "skipped" and "scanpp" in r2.reason
    assert _sha(p) == h1


def test_marker_alone_is_enough_to_skip(tmp_path):
    """A plain-JPEG ('e1') output has neither mask nor OCR layer; the marker is what keeps it from being re-encoded."""
    p = make_scan_pdf(tmp_path / "e1.pdf", [text_page(seed=3)])
    pipeline.process_file(p, Options(encoder="jpeg", jobs=1, backup=False, notify=False))
    with pikepdf.open(p) as pdf:
        ok, why = pdfio.classify_pdf(pdf)
        assert not ok and "already processed" in why
        # without the marker the plain-JPEG output would look like a fresh scan
        del pdf.docinfo["/ScanppVersion"]
        for pg in pdf.pages:
            del pg.obj["/ScanppPage"]
        assert pdfio.classify_pdf(pdf)[0]


# ----------------------------------------------------------------------------- paper boundary
def _page_with_backing(h_paper=900, backing=(196, 190, 176), shadow=True, text=True, w=850, h=1100):
    img = np.empty((h, w, 3), np.uint8)
    img[:] = backing
    img[:h_paper] = 252
    if shadow:
        img[h_paper - 3:h_paper + 3] = (25, 25, 25)
    if text:
        base = text_page(w=w, h=h_paper)
        img[:h_paper - 6] = base[:h_paper - 6]
    return img


def test_boundary_detected_and_painted():
    img = _page_with_backing()
    res = boundary.find(img, 100)
    assert [r["side"] for r in res] == ["B"]
    assert res[0]["depth_in"] == pytest.approx(2.0, abs=0.06)
    out, info = geometry.clean_edges(img, 100, fill=(255, 255, 255))
    assert (out[920:] == 255).all()                                           # everything beyond the edge is paper
    assert np.array_equal(out[:880], img[:880])                               # nothing above was touched
    assert info["boundary_B"] == pytest.approx(2.0, abs=0.06)


def test_boundary_tilted_edge_and_full_res():
    big = cv2.resize(_page_with_backing(), None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    M = cv2.getRotationMatrix2D((big.shape[1] / 2, big.shape[0] / 2), 0.7, 1.0)
    big = cv2.warpAffine(big, M, (big.shape[1], big.shape[0]), borderValue=(196, 190, 176))
    out, info = geometry.clean_edges(big, 300, fill=(255, 255, 255))
    assert "boundary_B" in info
    assert (out[-100:, 300:-300] == 255).all()
    k = int(big.shape[0] * 0.78)
    assert (out[300:k, 300:-300] == big[300:k, 300:-300]).all()                  # nothing on the paper was touched


@pytest.mark.parametrize("variant", ["black_bar", "black_bar_text", "blue_bar_text", "graph_to_edge", "text_to_edge",
                                     "full_page_text", "coloured_graphic", "blank_paper"])
def test_boundary_does_not_fire_on_content(variant, monkeypatch):
    img = text_page()
    h, w = img.shape[:2]
    if variant == "black_bar":
        img[:60] = 20
    elif variant == "black_bar_text":
        img[:60] = 20
        cv2.putText(img, "UNIT TEST", (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    elif variant == "blue_bar_text":
        img[:80] = (40, 90, 200)
        cv2.putText(img, "Chapter", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    elif variant == "graph_to_edge":
        for y in range(0, h, 10):
            img[y] = 150
        for x in range(0, w, 10):
            img[:, x] = 150
    elif variant == "text_to_edge":
        cv2.putText(img, "footer text right at the bottom", (60, h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    elif variant == "coloured_graphic":
        img[-120:] = (120, 200, 120)
        cv2.circle(img, (w // 2, h - 60), 40, (255, 255, 0), -1)
        cv2.rectangle(img, (100, h - 100), (300, h - 30), (0, 0, 255), -1)
    elif variant == "blank_paper":
        img[:] = 252
    out, info = geometry.clean_edges(img, 100, fill=(255, 255, 255))
    assert not [k for k in info if k.startswith("boundary_")], info
    with monkeypatch.context() as m:                          # identical to the pre-existing strip/coil cleaning alone
        m.setattr(boundary, "find", lambda *a, **k: [])
        ref, _ = geometry.clean_edges(img, 100, fill=(255, 255, 255))
    assert np.array_equal(out, ref)


def test_boundary_ink_beyond_edge_blocks_painting():
    img = _page_with_backing()
    cv2.putText(img, "NOTE", (200, 1020), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)      # ink on the 'backing'
    assert boundary.find(img, 100) == []


def test_boundary_dark_deep_bar_not_a_backing():
    assert boundary.find(_page_with_backing(backing=(15, 15, 15), shadow=False), 100) == []     # deep black block = content


def test_boundary_coloured_paper_light_backing():
    img = np.empty((1100, 850, 3), np.uint8)
    img[:] = (190, 225, 190)                                                   # green paper
    img[950:] = (236, 236, 232)                                                # lighter lid beyond the edge
    img[947:953] = (40, 60, 40)
    cv2.putText(img, "Graph the function", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (20, 20, 20), 2)
    res = boundary.find(img, 100)
    assert res and res[0]["side"] == "B"
