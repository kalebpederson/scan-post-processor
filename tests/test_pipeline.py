import cv2
import numpy as np
import pikepdf
import pytest
from PIL import Image

from helpers import DPI, blank_page, make_scan_pdf, text_page
from scanpp import cli, pdfio
from scanpp.pipeline import Options, process_array, process_file


def run(path, *extra):
    return cli.main(["application/pdf", "false", str(path), " ".join(("--jobs 1 --foreground",) + extra)])


def n_pages(p):
    with pikepdf.open(p) as pdf:
        return len(pdf.pages)


def test_blank_dropped_and_order_kept(tmp_path):
    pages = [text_page(seed=1), blank_page(), text_page(seed=2), blank_page(seed=5)]
    f = make_scan_pdf(tmp_path / "a.pdf", pages)
    assert run(f) == 0
    assert n_pages(f) == 2


def test_keep_blank_option(tmp_path):
    f = make_scan_pdf(tmp_path / "a.pdf", [text_page(), blank_page()])
    assert run(f, "--keep-blank") == 0
    assert n_pages(f) == 2


def test_all_blank_keeps_all(tmp_path):
    f = make_scan_pdf(tmp_path / "a.pdf", [blank_page(), blank_page(seed=3), blank_page(seed=4)])
    assert run(f) == 0
    assert n_pages(f) == 3


def test_single_page_parallel_jobs(tmp_path):
    f = make_scan_pdf(tmp_path / "a.pdf", [text_page()])
    assert cli.main([str(f), "--jobs", "8"]) == 0
    assert n_pages(f) == 1


def test_parallel_order_preserved(tmp_path):
    """Pages with distinct sizes: the output order must match the input order under real multiprocessing."""
    pages = [text_page(w=420 + 20 * i, h=500, lines=10, seed=i) for i in range(6)]
    f = make_scan_pdf(tmp_path / "o.pdf", pages, dpi=100)
    out = tmp_path / "o2.pdf"
    assert cli.main([str(f), "-o", str(out), "--jobs", "3", "--dpi", "100"]) == 0
    with pikepdf.open(out) as pdf:
        widths = [int(next(iter(p.Resources.XObject.values())).Width) for p in pdf.pages]
    assert widths == [420 + 20 * i for i in range(6)]
    assert not (tmp_path / ".originals").exists()      # -o does not touch the input


def test_gray_input_pdf(tmp_path):
    g = cv2.cvtColor(text_page(), cv2.COLOR_RGB2GRAY)
    f = make_scan_pdf(tmp_path / "g.pdf", [g], kind="jpeg")
    assert run(f) == 0
    with pikepdf.open(f) as pdf:
        im = next(iter(pdf.pages[0].Resources.XObject.values()))
        assert im.ColorSpace == "/DeviceGray"


def test_rotate90_preserved_and_content_upright(tmp_path):
    """A /Rotate 90 page: image is stored sideways; output keeps /Rotate and MediaBox, still deskews upright content."""
    upright = text_page(angle=1.5)
    stored = np.ascontiguousarray(np.rot90(upright, 1))      # rotate CCW so that /Rotate 90 (CW) displays it upright
    f = make_scan_pdf(tmp_path / "r.pdf", [stored], dpi=100, rotate=90)
    with pikepdf.open(f) as pdf:
        box0 = [float(v) for v in pdf.pages[0].MediaBox]
    out = tmp_path / "out.pdf"
    assert cli.main([str(f), "-o", str(out), "--jobs", "1", "--encoder", "jpeg", "--dpi", "100"]) == 0
    from scanpp.pipeline import _decode
    with pikepdf.open(out) as pdf:
        p = pdf.pages[0]
        assert int(p.get("/Rotate", 0)) == 90
        assert [float(v) for v in p.MediaBox] == pytest.approx(box0, abs=0.01)
        pi = pdfio.extract_page_input(p, 0)
    disp = np.rot90(_decode(pi), -1)                        # as displayed
    from scanpp.geometry import estimate_skew
    ang, _ = estimate_skew(cv2.cvtColor(np.ascontiguousarray(disp), cv2.COLOR_RGB2GRAY), 100)
    assert abs(ang) < 0.35


def test_mediabox_preserved_non_letter(tmp_path):
    f = make_scan_pdf(tmp_path / "a4.pdf", [text_page(w=827, h=1169)], dpi=100)
    with pikepdf.open(f) as pdf:
        b0 = [float(v) for v in pdf.pages[0].MediaBox]
    assert run(f) == 0
    with pikepdf.open(f) as pdf:
        assert [float(v) for v in pdf.pages[0].MediaBox] == pytest.approx(b0, abs=0.01)
        assert str(pdf.docinfo["/Title"]) == "a4"
        assert "scanpp" in str(pdf.docinfo["/Producer"]) and str(pdf.docinfo["/Creator"]) == "Simple Scan 48.1"


def test_downsample_to_output_dpi_and_no_upsample(tmp_path):
    rgb = text_page()
    out, d, st = process_array(rgb, 300.0 if False else 100.0, Options(dpi=50), allow_drop=False)
    assert out.shape[:2] == (550, 425) and d == 50
    out, d, st = process_array(rgb, 100.0, Options(dpi=200), allow_drop=False)
    assert out.shape == rgb.shape and d == 100          # never upsample


def test_stats_and_report(tmp_path):
    import json
    f = make_scan_pdf(tmp_path / "a.pdf", [text_page(angle=1.5), blank_page()])
    rep = tmp_path / "r.json"
    assert run(f, "--report", str(rep)) == 0
    data = json.loads(rep.read_text())
    pg = data["files"][0]["pages"]
    assert pg[0]["skew_applied"] and abs(pg[0]["skew_angle"] - 1.5) < 0.3
    assert pg[1]["dropped"] and pg[1]["blank_score"] < 40
    assert "bytes" in pg[0] and "t" in pg[0] and "edge_painted_frac" in pg[0]


@pytest.mark.parametrize("fmt,ext", [("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp")])
def test_image_inputs_in_place(tmp_path, fmt, ext):
    f = tmp_path / f"scan.{ext}"
    Image.fromarray(text_page(angle=1.0, paper=(240, 240, 225), noise=2)).save(f, fmt, dpi=(100, 100),
                                                                           **({"quality": 95} if fmt != "PNG" else {}))
    before = f.read_bytes()
    assert cli.main([f"image/{ext}", "false", str(f), "--jobs 1 --foreground"]) == 0
    with Image.open(f) as im:
        assert im.format == fmt
        arr = np.asarray(im.convert("RGB"))
    assert (arr[: arr.shape[0] // 20] >= 250).all()          # paper (top margin) is white now
    assert (tmp_path / ".originals" / f.name).read_bytes() == before


def test_blank_image_not_dropped(tmp_path):
    f = tmp_path / "b.png"
    Image.fromarray(blank_page()).save(f, dpi=(100, 100))
    assert cli.main([str(f)]) == 0
    assert f.exists()


def test_unsupported_file_untouched(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    assert cli.main([str(f)]) == 0 and f.read_text() == "hi"


def _word_page(depth=60, seed=3):
    """A blank page with ONE hand-written-looking word in light pencil (stroke depth `depth` gray levels, 200 dpi)."""
    h, w, dpi = 2200, 1700, 200
    img = np.full((h, w, 3), 250, np.uint8)
    img = np.clip(img + np.random.default_rng(seed).normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    rng = np.random.default_rng(7)
    x = 500
    for _ in range(6):                                        # six wobbly 0.3 in letter-like strokes, ~0.4 in tall
        pts = np.array([[x + i * 6, 900 + int(30 * np.sin(i / 3 + rng.uniform(0, 6))) + int(rng.integers(-6, 6))]
                        for i in range(14)], np.int32)
        cv2.polylines(mask, [pts], False, 255, 3, cv2.LINE_AA)
        cv2.line(mask, (x + 40, 860), (x + 44, 960), 255, 3, cv2.LINE_AA)
        x += 110
    a = cv2.GaussianBlur(mask, (0, 0), 0.8).astype(np.float32)[..., None] / 255
    return np.clip(img - a * depth, 0, 255).astype(np.uint8), dpi


def test_default_threshold_keeps_light_word_drops_dust():
    word, dpi = _word_page()
    o = Options()
    out, _, st = process_array(word.copy(), dpi, o)
    assert out is not None and not st["dropped"], st["blank_score"]
    assert 15 < st["blank_score"] < 400                       # kept, with margin over the default (word is ~25-110 in the corpus test)
    out, _, st = process_array(blank_page(), DPI, o)
    assert out is None and st["dropped"] and st["blank_score"] < 15


def test_threshold_high_drops_low_keeps():
    word, dpi = _word_page()
    score = process_array(word.copy(), dpi, Options(drop_blank=False))[2]["blank_score"]
    out, _, st = process_array(word.copy(), dpi, Options(blank_threshold=score * 2))
    assert out is None and st["dropped"]                      # high threshold: the word page counts as blank
    out, _, st = process_array(word.copy(), dpi, Options(blank_threshold=score / 2))
    assert out is not None and not st["dropped"]              # low threshold: kept
    out, _, st = process_array(blank_page(), DPI, Options(blank_threshold=0.0))
    assert out is not None                                    # 0 keeps everything


def test_threshold_through_cli_pdf_and_log(tmp_path, capfd):
    word, dpi = _word_page()
    pages = [text_page(seed=1), word, blank_page()]
    f = make_scan_pdf(tmp_path / "a.pdf", pages, dpi=dpi)
    g = tmp_path / "b.pdf"
    g.write_bytes(f.read_bytes())
    assert run(f) == 0                                        # default: word page kept, dust page dropped
    assert n_pages(f) == 2
    msgs = [ln for ln in capfd.readouterr().err.splitlines() if "dropped as blank" in ln]
    assert len(msgs) == 1 and "page 3" in msgs[0] and "blank_score" in msgs[0] and "threshold 15" in msgs[0]
    assert run(g, "--blank-threshold 100000") == 0            # everything scores below: all blank -> all kept
    assert n_pages(g) == 3
    h = tmp_path / "c.pdf"
    make_scan_pdf(h, pages, dpi=dpi)
    assert run(h, "--blank-threshold 0") == 0
    assert n_pages(h) == 3
