"""Duplex-aware bleed-through suppression: unit tests on a synthetic double-sided sheet, pipeline integration, safety nets."""
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pikepdf
import pytest

from helpers import H, W, blank_page, make_scan_pdf, text_page
from scanpp import cli, duplex, pdfio, pipeline
from scanpp.options import Options

GHOST = (205, 222, 255)         # pale blue: the marker's show-through
MARKER = (60, 110, 250)
BRO = Path(os.environ.get("SCANPP_BROTHER_RAW", "/nonexistent-brother-raw"))     # private Brother scans; tests skip when unset


def _strokes(rng, n, wmin=20, wmax=30):
    """n thick irregular marker strokes (bool mask)."""
    m = np.zeros((H, W), np.uint8)
    for _ in range(n):
        x, y = int(rng.integers(120, 650)), int(rng.integers(150, 950))
        pts = [(x, y)]
        for _ in range(int(rng.integers(2, 5))):
            x = int(np.clip(x + rng.integers(-160, 200), 80, 770))
            y = int(np.clip(y + rng.integers(-60, 60), 100, 1000))
            pts.append((x, y))
        cv2.polylines(m, [np.array(pts, np.int32)], False, 1, int(rng.integers(wmin, wmax)))
    return m > 0


def sheet(seed=5, front_seed=1, back_seed=2):
    """Front A with blue marker strokes; back B with its own green marker and a paler mirrored ghost of A's strokes.
    Returns A, B (clean back), ghost mask in B's frame."""
    rng = np.random.default_rng(seed)
    A = text_page(seed=front_seed, lines=30)
    B = text_page(seed=back_seed, lines=30)
    marker, own = _strokes(rng, 4, 16, 24), _strokes(rng, 3, 16, 24)
    A[marker] = MARKER
    B[own] = (70, 200, 90)
    gh = marker[:, ::-1]
    B[gh] = np.minimum(B[gh], np.array(GHOST, np.uint8))
    og = own[:, ::-1]
    A[og] = np.minimum(A[og], np.array((225, 255, 228), np.uint8))
    return A, B, gh


def with_pen(B):
    """B plus a dark red stroke crossing the band area and a dark green stroke elsewhere; returns the page and the two stroke masks."""
    B = B.copy()
    cv2.line(B, (100, 300), (700, 330), (150, 20, 20), 3)
    cv2.line(B, (100, 1040), (500, 1040), (20, 90, 30), 3)
    cross = np.zeros((H, W), np.uint8)
    cv2.line(cross, (100, 300), (700, 330), 1, 3)
    outside = np.zeros((H, W), np.uint8)
    cv2.line(outside, (100, 1040), (500, 1040), 1, 3)
    return B, cross > 0, outside > 0


def inked(img):
    return img.min(-1) < 250


def content(path):
    """Everything that makes up the pages of a PDF (content streams, image streams and their dictionaries, masks), without the document
    metadata: scanpp output differs run to run in its timestamps and /ID, never in the pages."""
    out = []
    with pikepdf.open(str(path)) as pdf:
        for page in pdf.pages:
            ent = [page.obj.MediaBox.unparse(), bytes(page.Contents.read_bytes()) if isinstance(page.Contents, pikepdf.Stream) else b""]
            for name, xo in sorted(page.obj.Resources.get("/XObject", {}).items()):
                ent.append((name, xo.unparse(), bytes(xo.read_raw_bytes())))
                for k in ("/Mask", "/SMask"):
                    if k in xo:
                        ent.append((k, bytes(xo[k].read_raw_bytes())))
            out.append(ent)
    return out


def tasks_of(path):
    with pikepdf.open(str(path)) as pdf:
        return list(pdfio.iter_page_inputs(pdf))


def clean_both(tasks, plan, opts, which, post=True):
    """Cleaned page `which` as the pipeline produces it, with or without the duplex hook."""
    t = tasks[which]
    rgb = pipeline._decode(t)
    hook = pipeline._duplex_post(tasks[plan[which].partner], plan[which], opts, which + 1) if post else None
    return pipeline.process_array(rgb, t.dpi, opts, allow_drop=False, resize=False, post_clean=hook)


@pytest.fixture(scope="module")
def pen_sheet(tmp_path_factory):
    A, B, gh = sheet()
    Bp, cross, outside = with_pen(B)
    path = make_scan_pdf(tmp_path_factory.mktemp("sheet") / "s.pdf", [A, Bp], kind="flate")     # lossless: exact pixel checks
    return dict(path=path, A=A, B=B, Bp=Bp, ghost=gh, cross=cross, outside=outside)


OPTS = Options(jobs=1, deskew=False)


def test_pair_accepted_ghost_removed_strokes_kept(pen_sheet):
    tasks = tasks_of(pen_sheet["path"])
    plan, t_maps = pipeline.plan_duplex(iter(tasks), 2, OPTS)
    assert set(plan) == {0, 1} and plan[1].partner == 0 and plan[0].partner == 1 and set(t_maps) == {0, 1}
    off, _, _ = clean_both(tasks, plan, OPTS, 1, post=False)
    on, _, st = clean_both(tasks, plan, OPTS, 1)
    assert st["duplex_partner"] == 0 and st["duplex_ncc"] > 0.5 and st["duplex_removed_px"] > 10000
    assert "duplex_supp" in st["t"]
    # ghost px that the single-page cleaner leaves; ghost within 8 px of B's own dark ink is protected by design and not counted
    g = pen_sheet["ghost"] & (np.abs(pen_sheet["B"].astype(int) - GHOST).max(-1) <= 2)
    g &= ~cv2.dilate(((pen_sheet["Bp"].min(-1) < 150)).astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
    g &= inked(off)
    assert g.sum() > 2000
    assert (g & ~inked(on)).sum() > 0.9 * g.sum()
    for name in ("cross", "outside"):
        k = pen_sheet[name] & (off.astype(int).sum(-1) < 450)         # the dark core of the stroke
        assert k.sum() > 500
        assert (k & inked(on)).sum() >= 0.99 * k.sum(), name
    # the front gets the same treatment from the back (its pale green ghost); no crash, only ink removed
    fa, _, sta = clean_both(tasks, plan, OPTS, 0)
    assert sta["duplex_partner"] == 1
    fo, _, _ = clean_both(tasks, plan, OPTS, 0, post=False)
    assert not (inked(fa) & ~inked(fo)).any()


def test_unrelated_pages_not_paired_and_bytes_identical(tmp_path):
    A, _, _ = sheet(seed=5, front_seed=1)
    _, C, _ = sheet(seed=77, back_seed=3)
    f = make_scan_pdf(tmp_path / "u.pdf", [A, C, text_page(seed=9)])
    maps = [duplex.ink_maps(p, 100.0) for p in (A, C)]
    assert duplex.pair_sheets(maps) == []
    for mode in ("auto", "off"):
        (tmp_path / mode).mkdir()
        assert cli.main(["--jobs", "1", "--foreground", "--duplex", mode, str(f), "-o", str(tmp_path / mode / "out.pdf")]) == 0
    assert content(tmp_path / "auto" / "out.pdf") == content(tmp_path / "off" / "out.pdf")


def _run(tmp_path, f, *extra, name="out.pdf"):
    """Process f to tmp_path/<name> (the title metadata follows the output name, so compare outputs of equal names in different dirs)."""
    (tmp_path / name[:-4]).mkdir(exist_ok=True)
    out = tmp_path / name[:-4] / "out.pdf"
    rep = tmp_path / (name + ".json")
    assert cli.main(["--jobs", "1", "--foreground", "--report", str(rep), *extra, str(f), "-o", str(out)]) == 0
    return json.loads(rep.read_text())["files"][0]["pages"]


def test_file_with_pair_uses_it_and_odd_page_count(tmp_path):
    A, B, _ = sheet()
    f = make_scan_pdf(tmp_path / "odd.pdf", [A, B, text_page(seed=4)])
    pg = _run(tmp_path, f, name="a.pdf")
    assert [p["duplex_partner"] for p in pg] == [1, 0, None]
    assert pg[0]["duplex_removed_px"] > 0 and pg[1]["duplex_removed_px"] > 0 and pg[2]["duplex_removed_px"] == 0
    assert all("duplex_maps" in p["t"] for p in pg) and pg[1]["t"]["duplex_supp"] >= 0
    off = _run(tmp_path, f, "--duplex", "off", name="o.pdf")
    assert [p["duplex_partner"] for p in off] == [None] * 3 and "duplex_maps" not in off[0]["t"]
    with pikepdf.open(str(tmp_path / "a" / "out.pdf")) as pdf:
        assert len(pdf.pages) == 3
    ca, co = content(tmp_path / "a" / "out.pdf"), content(tmp_path / "o" / "out.pdf")
    assert any(len(e) > 2 for e in ca) and ca[:2] != co[:2] and ca[2] == co[2]      # paired pages change, the unpaired one does not


def test_single_page_pdf_skips_duplex(tmp_path, monkeypatch):
    monkeypatch.setattr(duplex, "ink_maps", lambda *a, **k: pytest.fail("duplex analysis ran on a 1-page PDF"))
    A, _, _ = sheet()
    f = make_scan_pdf(tmp_path / "one.pdf", [A])
    pg = _run(tmp_path, f)
    assert pg[0]["duplex_partner"] is None


def test_image_file_skips_duplex(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.setattr(duplex, "ink_maps", lambda *a, **k: pytest.fail("duplex analysis ran on an image"))
    A, _, _ = sheet()
    p = tmp_path / "a.png"
    Image.fromarray(A).save(p, dpi=(100, 100))
    assert cli.main(["--jobs", "1", "--foreground", "--no-backup", str(p), "-o", str(tmp_path / "o.png")]) == 0


def test_blank_back_no_pair_and_dropped(tmp_path):
    A, _, _ = sheet()
    f = make_scan_pdf(tmp_path / "bb.pdf", [A, blank_page()])
    pg = _run(tmp_path, f)
    assert pg[0]["duplex_partner"] is None and pg[1]["dropped"]
    off = _run(tmp_path, f, "--duplex", "off", name="o.pdf")
    assert content(tmp_path / "out" / "out.pdf") == content(tmp_path / "o" / "out.pdf")


def test_blank_back_with_ghost_is_dropped_source_still_valid(tmp_path):
    """A back that is blank apart from the ghost: still dropped; the pair (if any) must not break the front."""
    A, _, _ = sheet()
    Bg = blank_page(w=W, h=H)
    gh = _strokes(np.random.default_rng(5), 4, 16, 24)[:, ::-1]
    Bg[gh] = np.minimum(Bg[gh], np.array((240, 244, 255), np.uint8))
    f = make_scan_pdf(tmp_path / "bg.pdf", [A, Bg])
    pg = _run(tmp_path, f, "--blank-threshold", "30")
    assert pg[1]["dropped"] and "duplex_supp" not in pg[1]["t"] and "duplex_maps" in pg[1]["t"]
    assert pg[0]["duplex_removed_px"] == 0


def test_black_print_only_not_paired():
    A = text_page(seed=1)
    B = text_page(seed=2)
    ghost = np.clip(A[:, ::-1].astype(int) + 180, 0, 255).astype(np.uint8)       # pale grey mirrored print
    B = np.minimum(B, ghost)
    assert duplex.pair_sheets([duplex.ink_maps(A, 100.0), duplex.ink_maps(B, 100.0)]) == []


def test_safety_cap_discards_mask(pen_sheet, monkeypatch, caplog):
    monkeypatch.setattr(logging.getLogger("scanpp"), "propagate", True)
    tasks = tasks_of(pen_sheet["path"])
    plan, _ = pipeline.plan_duplex(iter(tasks), 2, OPTS)
    monkeypatch.setattr(duplex, "bleed_suppression", lambda s, d, r, **k: np.ones(d.shape[:2], np.float32))     # would erase the whole page
    with caplog.at_level(logging.INFO, logger="scanpp"):
        on, _, st = clean_both(tasks, plan, OPTS, 1)
    off, _, _ = clean_both(tasks, plan, OPTS, 1, post=False)
    assert np.array_equal(on, off)
    assert st["duplex_partner"] is None and st["duplex_removed_px"] == 0 and st["duplex_discarded_px"] > 0
    assert "discarded" in caplog.text


def test_cleaned_page_early_out_is_exact(pen_sheet, monkeypatch):
    """`bleed_suppression(cleaned=...)` may skip the destination analysis only where the blend would change nothing."""
    tasks = tasks_of(pen_sheet["path"])
    plan, _ = pipeline.plan_duplex(iter(tasks), 2, OPTS)
    src, dst = (pipeline._upright_edge_clean(tasks[i], OPTS) for i in (0, 1))
    off, _, _ = clean_both(tasks, plan, OPTS, 1, post=False)
    on, _, _ = clean_both(tasks, plan, OPTS, 1)
    white = np.full_like(off, 255)
    full = duplex.bleed_suppression(src, dst, plan[1].reg)
    assert full.any()
    # a cleaned page that still carries ink under the footprint: the same mask (nothing skipped)
    assert np.array_equal(duplex.bleed_suppression(src, dst, plan[1].reg, cleaned=off), full)
    for cleaned in (off, on, white):
        s = duplex.bleed_suppression(src, dst, plan[1].reg, cleaned=cleaned)
        assert np.array_equal(duplex.apply_suppression(cleaned, s), duplex.apply_suppression(cleaned, full))
    # a cleaned page that is paper everywhere near the footprint: no analysis of the destination at all
    monkeypatch.setattr(duplex, "_native_density", lambda *a, **k: pytest.fail("destination analysed"))
    assert not duplex.bleed_suppression(src, dst, plan[1].reg, cleaned=white).any()
    tint = np.full_like(off, (250, 245, 235))
    assert not duplex.bleed_suppression(src, dst, plan[1].reg, cleaned=tint, paper_rgb=(250, 245, 235)).any()


def test_source_without_marker_skips_destination_analysis(monkeypatch):
    reg = duplex.PairReg(np.array([[-1.0, 0, W - 1], [0, 1, 0]]), 100.0, 100.0)
    monkeypatch.setattr(duplex, "_native_density", lambda *a, **k: pytest.fail("destination analysed"))
    s = duplex.bleed_suppression(text_page(seed=1), text_page(seed=2), reg)
    assert s.shape == (H, W) and not s.any()


def test_pair_test_skips_registration_without_colour(monkeypatch):
    A, B, _ = sheet()
    mA, mB = duplex.ink_maps(A, 100.0), duplex.ink_maps(B, 100.0)
    assert duplex.test_pair(mA, mB) is not None
    assert duplex.test_pair(mA, None) is None
    plain = [duplex.ink_maps(text_page(seed=i), 100.0) for i in (1, 2)]
    monkeypatch.setattr(duplex, "register_pair", lambda *a, **k: pytest.fail("registered two pages without colour ink"))
    assert duplex.test_pair(*plain) is None


def test_keeps_pair_conflict_resolution():
    feat = lambda sc: dict(ncc=sc, ga=1.0, gb=0.5)  # noqa: E731
    cand = {0: feat(0.4), 1: feat(0.5), 2: feat(0.9), 5: feat(0.5)}
    # pairs 0, 1 and 2 touch: 0 (0.4) does not beat 1 (0.5) by 1.5x, nor does 1 beat 2 (0.9); 2 beats 1; 5 has no rival
    assert [duplex.keeps_pair(cand, i) for i in (0, 1, 2, 5)] == [False, False, True, True]


def test_parallel_pipeline_matches_serial(tmp_path):
    """Several sheets, pages in between, 2 workers: the streamed analysis plans exactly what the serial one does and the output is the same."""
    A, B, _ = sheet()
    A2, B2, _ = sheet(seed=8, front_seed=3, back_seed=4)
    f = make_scan_pdf(tmp_path / "m.pdf", [text_page(seed=6), A, B, text_page(seed=7), A2, B2, text_page(seed=9)], kind="flate")
    serial = _run(tmp_path, f, name="s.pdf")
    par = _run(tmp_path, f, "--jobs", "2", name="p.pdf")
    assert [p["duplex_partner"] for p in serial] == [None, 2, 1, None, 5, 4, None]
    assert [p["duplex_partner"] for p in par] == [p["duplex_partner"] for p in serial]
    assert all("duplex_maps" in p["t"] for p in par)
    assert content(tmp_path / "s" / "out.pdf") == content(tmp_path / "p" / "out.pdf")


def test_exception_in_suppression_page_still_processed(tmp_path, pen_sheet, monkeypatch, capfd):

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(duplex, "bleed_suppression", boom)
    pg = _run(tmp_path, pen_sheet["path"])
    assert "duplex suppression failed (RuntimeError: synthetic failure)" in capfd.readouterr().err
    assert [p["duplex_partner"] for p in pg] == [None, None] and all("duplex_error" in p for p in pg)
    off = _run(tmp_path, pen_sheet["path"], "--duplex", "off", name="o.pdf")
    assert content(tmp_path / "out" / "out.pdf") == content(tmp_path / "o" / "out.pdf")


def test_exception_in_analysis_pages_still_processed(tmp_path, pen_sheet, monkeypatch, capfd):
    monkeypatch.setattr(duplex, "ink_maps", lambda *a, **k: (_ for _ in ()).throw(ValueError("no maps")))
    pg = _run(tmp_path, pen_sheet["path"])
    assert "duplex analysis failed" in capfd.readouterr().err and len(pg) == 2 and not any(p["dropped"] for p in pg)


def test_paper_keep_uses_page_paper_colour(tmp_path):
    A, B, _ = sheet()
    tint = np.array((250, 245, 235), np.float32)
    A, B = [np.clip(p.astype(np.float32) * tint / 255, 0, 255).astype(np.uint8) for p in (A, B)]
    f = make_scan_pdf(tmp_path / "k.pdf", [A, B], kind="flate")
    pg = _run(tmp_path, f, "--paper", "keep")
    assert pg[1]["duplex_partner"] == 0 and pg[1]["duplex_removed_px"] > 0


def test_cli_option():
    ns, _ = cli.parse_args(["--duplex", "off", "a.pdf"])
    assert cli.options_from(ns).duplex == "off"
    ns, _ = cli.parse_args(["a.pdf"])
    assert cli.options_from(ns).duplex == "auto"
    ns, keep = cli.parse_args(["application/pdf", "false", "/x/a.pdf", "--duplex off --jobs 2"])
    assert cli.options_from(ns).duplex == "off"
    with pytest.raises(SystemExit):
        cli.parse_args(["--duplex", "maybe", "a.pdf"])


def test_duplex_version_in_report(tmp_path):
    A, _, _ = sheet()
    f = make_scan_pdf(tmp_path / "v.pdf", [A])
    rep = tmp_path / "r.json"
    assert cli.main(["--jobs", "1", "--foreground", "--report", str(rep), str(f), "-o", str(tmp_path / "o.pdf")]) == 0
    d = json.loads(rep.read_text())
    assert d["duplex_version"] == duplex.DUPLEX_VERSION and d["options"]["duplex"] == "auto"


# ----------------------------------------------------------------------------- real scan (Brother sheet 1)
@pytest.mark.skipif(not (BRO / "page002.tif").exists(), reason="Brother scans not present")
def test_brother_sheet1_marker_bands(tmp_path):
    from PIL import Image
    A, B = [np.ascontiguousarray(np.array(Image.open(BRO / f"page{i:03d}.tif").convert("RGB"))) for i in (1, 2)]
    f = make_scan_pdf(tmp_path / "br.pdf", [A, B], dpi=300, kind="jpeg")
    pg = _run(tmp_path, f, "--jobs", "2", name="a.pdf")
    assert pg[1]["duplex_partner"] == 0 and pg[0]["duplex_partner"] == 1
    # marker-band dark pixels on the back, as in research/r5_duplex/eval_brother1.py (REPORT: 15,113 -> 967 px > 10 gl)
    tasks = tasks_of(f)
    opts = Options(jobs=1, deskew=False)
    plan, _ = pipeline.plan_duplex(iter(tasks), 2, opts)
    assert plan
    rB = pipeline._upright_edge_clean(tasks[1], opts)
    src = pipeline._upright_edge_clean(tasks[0], opts)
    rA = duplex._source_in_dst_frame(src, plan[1].reg, rB.shape[:2])
    dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2])
    cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
    bl = cv2.dilate(((cA > 0.3) & (rA[0] > rA[2] + 0.15)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
    gray = lambda o: 255 - o.astype(np.float32) @ np.array([.299, .587, .114], np.float32)  # noqa: E731
    off, _, _ = clean_both(tasks, plan, opts, 1, post=False)
    on, _, st = clean_both(tasks, plan, opts, 1)
    n_off, n_on = int((gray(off)[bl] > 10).sum()), int((gray(on)[bl] > 10).sum())
    assert n_off > 8000 and n_on < 0.2 * n_off, (n_off, n_on)
    assert st["duplex_removed_px"] > 5000
