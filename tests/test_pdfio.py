import numpy as np
import pikepdf
import pytest

from helpers import DPI, blank_page, make_digital_pdf, make_scan_pdf, text_page
from scanpp import pdfio
from scanpp.encode import EncodedPage, encode_page, has_color


def test_dpi_detection(tmp_path):
    p = make_scan_pdf(tmp_path / "a.pdf", [text_page()], dpi=100)
    with pikepdf.open(p) as pdf:
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
    assert pi.kind == "jpeg" and pi.size == (850, 1100)
    assert pi.dpi == pytest.approx(100, abs=0.01)
    assert pdfio.detect_dpi(2550, 3300, (0, 0, 612, 792)) == pytest.approx(300)
    assert pdfio.detect_dpi(1698, 2197, (0, 0, 611.28, 790.92)) == pytest.approx(200, abs=0.1)


def test_scan_pdf_is_classified_scan(tmp_path):
    p = make_scan_pdf(tmp_path / "a.pdf", [text_page(), text_page()])
    with pikepdf.open(p) as pdf:
        assert pdfio.classify_pdf(pdf) == (True, "ok")


def test_digital_pdf_is_not_scan(tmp_path):
    with pikepdf.open(make_digital_pdf(tmp_path / "d.pdf")) as pdf:
        ok, why = pdfio.classify_pdf(pdf)
    assert not ok and "font" in why


def test_text_operator_over_image_is_not_scan(tmp_path):
    """OCR'd PDF: image + invisible text must not be rewritten."""
    p = make_scan_pdf(tmp_path / "o.pdf", [text_page()])
    with pikepdf.open(p, allow_overwriting_input=True) as pdf:
        page = pdf.pages[0]
        page.Contents = pikepdf.Stream(pdf, page.Contents.read_bytes() + b"\nBT 3 Tr (x) Tj ET")
        ok, why = pdfio.classify_pdf(pdf)
    assert not ok and "operator" in why


def test_two_images_not_scan(tmp_path):
    p = make_scan_pdf(tmp_path / "t.pdf", [text_page()])
    with pikepdf.open(p, allow_overwriting_input=True) as pdf:
        page = pdf.pages[0]
        page.Resources.XObject["/Im1"] = page.Resources.XObject["/Im0"]
        assert not pdfio.classify_pdf(pdf)[0]


@pytest.mark.parametrize("kind", ["flate", "flate_gray", "cmyk", "bilevel", "runlength"])
def test_extract_other_image_types(tmp_path, kind):
    page = text_page(w=200, h=260)
    p = make_scan_pdf(tmp_path / "k.pdf", [page], dpi=50, kind=kind)
    with pikepdf.open(p) as pdf:
        assert pdfio.classify_pdf(pdf)[0]
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
    assert pi.kind == "raw" and pi.size == (200, 260)
    from scanpp.pipeline import _decode
    arr = _decode(pi)
    assert arr.shape == (260, 200, 3)
    if kind in ("flate", "runlength"):
        assert np.array_equal(arr, page)
    elif kind != "bilevel":
        assert np.abs(arr.astype(int) - page).mean() < 8


def test_rotate_and_mediabox_read(tmp_path):
    p = make_scan_pdf(tmp_path / "r.pdf", [text_page(w=300, h=200)], dpi=100, rotate=90)
    with pikepdf.open(p) as pdf:
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
    assert pi.rotate == 90 and pi.media_box == pytest.approx((0, 0, 216, 144))


def test_metadata():
    m = pdfio.build_metadata({"/Creator": "Simple Scan 48.1", "/Producer": "cairo"}, "My Doc")
    assert m["title"] == "My Doc" and m["creator"] == "Simple Scan 48.1"
    assert m["producer"].startswith("cairo; scanpp")
    assert pdfio.build_metadata({}, "x")["producer"].startswith("scanpp")
    assert pdfio.build_metadata({"/Producer": "scanpp 0.1"}, "x")["producer"] == "scanpp 0.1"


def test_write_pdf_roundtrip(tmp_path):
    pages = []
    first = text_page(w=200, h=260)
    first[50:60, 20:150] = 0
    for arr in (first, np.full((260, 200, 3), 255, np.uint8)):
        ep = encode_page(arr, 100, (144.0, 187.2), encoder="jpeg")
        ep.media_box = (0.0, 0.0, 144.0, 187.2)
        pages.append(ep)
    out = tmp_path / "o.pdf"
    pdfio.write_pdf(pages, out, {"title": "T", "producer": "P"})
    pdfio.validate_pdf(out, 2)
    with pytest.raises(ValueError):
        pdfio.validate_pdf(out, 3)
    with pikepdf.open(out) as pdf:
        assert str(pdf.docinfo["/Title"]) == "T"
        assert pdfio.classify_page(pdf.pages[0])[0]
        assert pdf.pages[1].Contents.read_bytes() == b""      # blank page: empty content stream


def test_gray_vs_color_decision():
    gray_page = text_page()
    assert not has_color(gray_page)
    col = gray_page.copy()
    col[100:140, 100:400] = (255, 0, 0)
    assert has_color(col)
    assert encode_page(gray_page, 100, (612, 792), encoder='jpeg').xobjects["Im0"].colorspace == "DeviceGray"
    assert encode_page(col, 100, (612, 792), encoder='jpeg').xobjects["Im0"].colorspace == "DeviceRGB"


def test_unknown_encoder():
    with pytest.raises(ValueError):
        encode_page(text_page(), 100, (612, 792), encoder="nope")
