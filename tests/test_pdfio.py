import numpy as np
import pikepdf
import pytest
from pikepdf import Name

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


# ----------------------------------------------------------------------------- legacy /SMask handling
def _with_smask(tmp_path, alpha=None, kind="runlength", extra=None, wrap=False, size=(200, 260)):
    """Scan PDF whose image carries an 8-bit gray /SMask (alpha, default all 255) or other mask keys."""
    page = text_page(w=size[0], h=size[1])
    p = make_scan_pdf(tmp_path / "m.pdf", [page], dpi=50, kind=kind)
    with pikepdf.open(p, allow_overwriting_input=True) as pdf:
        img = pdf.pages[0].Resources.XObject.Im0
        if alpha is not None:
            sm = pikepdf.Stream(pdf, alpha.tobytes())
            sm.Type, sm.Subtype, sm.Width, sm.Height = Name.XObject, Name.Image, alpha.shape[1], alpha.shape[0]
            sm.ColorSpace, sm.BitsPerComponent = Name.DeviceGray, 8
            img.SMask = sm
        for k, v in (extra or {}).items():
            img[k] = v(pdf) if callable(v) else v
        if wrap:   # repaired legacy file: extra scale cm, MediaBox Letter
            pg = pdf.pages[0]
            pg.MediaBox = [0, 0, 612, 792]
            pg.Contents = pikepdf.Stream(pdf, b"q 2.7778 0 0 2.7778 0 0 cm\n" + pg.Contents.read_bytes() + b"\nQ")
        pdf.save(str(p))
    return p


def _classify(p):
    with pikepdf.open(p) as pdf:
        return pdfio.classify_pdf(pdf)


def test_opaque_smask_is_plain_scan(tmp_path):
    p = _with_smask(tmp_path, np.full((260, 200), 255, np.uint8))
    assert _classify(p) == (True, "ok")
    with pikepdf.open(p) as pdf:
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
    assert pi.kind == "raw" and pi.size == (200, 260)


def test_smask_with_one_pixel_border_is_plain_scan(tmp_path):
    a = np.full((1300, 1000), 255, np.uint8)      # 1-px border must stay below 0.5 % of the pixels
    a[0, :] = a[-1, :] = 0
    a[:, 0] = a[:, -1] = 128
    assert _classify(_with_smask(tmp_path, a, size=(1000, 1300)))[0]


def test_smask_with_interior_transparency_is_skipped(tmp_path):
    a = np.full((260, 200), 255, np.uint8)
    a[100:102, 100:102] = 0          # tiny (< 0.5 %) but in the interior
    ok, why = _classify(_with_smask(tmp_path, a))
    assert not ok and "mask" in why
    b = np.full((260, 200), 255, np.uint8)
    b[:, :100] = 0                    # large transparent area
    assert not _classify(_with_smask(tmp_path, b))[0]


def test_explicit_mask_still_skipped(tmp_path):
    def stencil(pdf):
        m = pikepdf.Stream(pdf, bytes(33 * 260))
        m.Type, m.Subtype, m.Width, m.Height = Name.XObject, Name.Image, 200, 260
        m.ImageMask, m.BitsPerComponent = True, 1
        return m
    ok, why = _classify(_with_smask(tmp_path, None, extra={"/Mask": stencil}))
    assert not ok and "mask" in why
    # opaque SMask next to an explicit /Mask is still a masked image
    a = np.full((260, 200), 255, np.uint8)
    assert not _classify(_with_smask(tmp_path, a, extra={"/Mask": stencil}))[0]


def test_smask_in_data_skipped(tmp_path):
    a = np.full((260, 200), 255, np.uint8)
    assert not _classify(_with_smask(tmp_path, a, extra={"/SMaskInData": 1}))[0]


def test_runlength_image_with_smask_extracts_pixels(tmp_path):
    page = text_page(w=200, h=260)
    p = _with_smask(tmp_path, np.full((260, 200), 255, np.uint8))
    with pikepdf.open(p) as pdf:
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
    arr = np.frombuffer(pi.data, np.uint8).reshape(260, 200, 3)
    assert np.array_equal(arr, page)


def test_wrapped_cm_uses_mediabox_for_dpi_and_output_is_letter(tmp_path):
    from scanpp.pipeline import Options, process_file
    page = text_page(w=2544, h=3292)
    p = make_scan_pdf(tmp_path / "w.pdf", [page], dpi=864, kind="runlength")   # original 220x285 pt style geometry
    with pikepdf.open(p, allow_overwriting_input=True) as pdf:
        pg = pdf.pages[0]
        pg.MediaBox = [0, 0, 612, 792]
        pg.Contents = pikepdf.Stream(pdf, b"q 2.7778 0 0 2.7778 0 0 cm\n" + pg.Contents.read_bytes() + b"\nQ")
        sm = pikepdf.Stream(pdf, np.full((3292, 2544), 255, np.uint8).tobytes())
        sm.Type, sm.Subtype, sm.Width, sm.Height = Name.XObject, Name.Image, 2544, 3292
        sm.ColorSpace, sm.BitsPerComponent = Name.DeviceGray, 8
        pg.Resources.XObject.Im0.SMask = sm
        pdf.save(str(p))
    with pikepdf.open(p) as pdf:
        assert pdfio.classify_pdf(pdf) == (True, "ok")
        pi = pdfio.extract_page_input(pdf.pages[0], 0)
    assert 298.5 < pi.dpi < 299.5
    res = process_file(p, Options(backup=False, jobs=1, notify=False))
    assert res.status == "processed", res.reason
    with pikepdf.open(p) as out:
        mb = [float(v) for v in out.pages[0].MediaBox]
        assert mb == pytest.approx([0, 0, 612, 792], abs=1.0)
        for obj in out.pages[0].Resources.XObject.values():
            assert "/SMask" not in obj
