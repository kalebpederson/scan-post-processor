"""Fold-crease detection and removal (src/scanpp/fold.py, research/r6_fold/REPORT.md).  All pages are synthetic: a procedural crease
(thin broken gray line with a mottled bluish shadow band) on generated pages."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pytest

from scanpp import cli, fold, pipeline
from scanpp.options import STAGES, Options

DPI = 150
WORDS = ["Algebra", "triangle", "the", "measure", "of", "angle", "ABC", "equals", "30", "degrees"]


def paper(w_in=8.5, h_in=11.0, dpi=DPI, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((int(h_in * dpi), int(w_in * dpi), 3), 251.0, np.float32)
    return np.clip(img + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)


def text_page(dpi=DPI, gap=(0.42, 0.58), seed=0):
    """Text lines with a free band around the middle, except one short line that the half-height crease runs through."""
    rng = np.random.default_rng(seed)
    img = paper(dpi=dpi, seed=seed)
    h, w = img.shape[:2]
    cy = h // 2
    sc = dpi / 100 * 0.6
    y = int(0.8 * dpi)
    while y < h - 0.8 * dpi:
        crossing = abs(y - cy - 4) <= 8
        if gap[0] * h < y < gap[1] * h and not crossing:
            y += int(0.34 * dpi)
            continue
        x, xmax = int(0.7 * dpi), (int(3.2 * dpi) if crossing else w - int(0.9 * dpi))
        while x < xmax:
            wd = WORDS[rng.integers(len(WORDS))]
            cv2.putText(img, wd, (x, y), cv2.FONT_HERSHEY_SIMPLEX, sc, (20, 20, 20), 1, cv2.LINE_AA)
            x += int(cv2.getTextSize(wd, cv2.FONT_HERSHEY_SIMPLEX, sc, 1)[0][0]) + int(0.14 * dpi)
        y += int(0.34 * dpi)
    return img


def crease(img, y_frac=0.5, tilt=0.3, depth=50, band=3.5, seed=1, dpi=DPI):
    """A fold line: thin (1-3 px) broken dark line, a bluish shadow band 0.2 in wide on one side, slight tilt."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    x = np.arange(w)
    yc = y_frac * h + np.tan(np.radians(tilt)) * (x - w / 2)
    d = np.arange(h)[:, None] - yc[None, :]
    core = np.exp(-0.5 * (d / (0.008 * dpi)) ** 2)
    alive = np.convolve((rng.random(w) > 0.13).astype(float), np.ones(9) / 9, "same")[None, :] > 0.5
    shadow = np.where((d > 0) & (d < 0.2 * dpi), 1 - d / (0.2 * dpi), 0)
    mott = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.03 * dpi)
    sub = depth * core * alive + band * shadow * (1 + 0.5 * mott)
    out = img.astype(np.float32) - sub[..., None] * np.array([1.0, 1.0, 0.85], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def band_px(img, lo=4.0, hi=7.5, y_frac=0.5, half_in=0.04, level=240):
    """Pixels darker than `level` in the text-free columns around the crease row."""
    h, w = img.shape[:2]
    r = int(half_in * DPI)
    y = int(y_frac * h)
    return int((img[y - r:y + r, int(lo * DPI):int(hi * DPI)].min(-1) < level).sum())


@pytest.fixture(scope="module")
def creased():
    return crease(text_page())


def test_crease_detected_and_removed_text_kept(creased):
    folds = fold.detect_folds(creased, DPI)
    assert len(folds) == 1 and folds[0].info["tier"] == "A"
    assert abs(folds[0].y_frac - 0.5) < 0.01
    mask = fold.fold_mask(creased, DPI, folds)
    out = fold.remove_folds(creased, DPI, folds, mask=mask)
    assert out is not creased and creased.dtype == out.dtype
    dark = creased.min(-1) < 120
    assert dark.sum() > 2000
    assert (dark & (out.min(-1) < 120)).sum() >= 0.99 * dark.sum()               # text crossing the crease is kept
    before, after = band_px(creased), band_px(out)
    assert before > 1500 and after < 0.05 * before                                # the crease is gone where there is no text
    rows = np.flatnonzero(mask.any(1))
    assert abs(rows.min() / creased.shape[0] - 0.5) < 0.03 and rows.max() - rows.min() < 0.3 * DPI
    assert (out == creased)[: rows.min() - 20].all() and (out == creased)[rows.max() + 20:].all()      # nothing else touched


def test_remove_inplace(creased):
    folds = fold.detect_folds(creased, DPI)
    ref = fold.remove_folds(creased, DPI, folds)
    work = creased.copy()
    assert fold.remove_folds(work, DPI, folds, inplace=True) is work
    assert np.array_equal(work, ref)


def test_pipeline_removes_crease_and_reports(creased):
    opts = Options(deskew=False)
    on, _, st = pipeline.process_array(creased.copy(), DPI, opts, allow_drop=False, resize=False)
    off, _, st_off = pipeline.process_array(creased.copy(), DPI, Options(deskew=False, folds="off"), allow_drop=False, resize=False)
    assert len(st["folds"]) == 1 and set(st["folds"][0]) == {"row_frac", "angle", "tier", "score"}
    assert st["folds"][0]["tier"] == "A" and abs(st["folds"][0]["row_frac"] - 0.5) < 0.01
    assert st["fold_px_removed"] > 1000 and st["t"]["fold_detect"] > 0 and st["t"]["fold_remove"] > 0
    assert st_off["folds"] == [] and st_off["fold_px_removed"] == 0 and "fold_detect" not in st_off["t"]
    assert {"fold_detect", "fold_remove"} <= set(STAGES)
    assert band_px(on, level=250) < 0.2 * band_px(off, level=250)                 # the cleaner would have kept the crease


def test_pipeline_leaves_the_input_alone_when_read_only(creased):
    ro = creased.copy()
    ro.flags.writeable = False
    out, _, st = pipeline.process_array(ro, DPI, Options(deskew=False, edge_clean=False), allow_drop=False, resize=False)
    assert st["folds"] and np.array_equal(ro, creased)


def test_ruled_notebook_page_not_a_fold():
    img = text_page(gap=(0, 0))
    for y in np.arange(0.9 * DPI, 10.5 * DPI, 0.31 * DPI).astype(int):          # periodic faint blue rules, edge to edge
        img[y:y + 2] = np.minimum(img[y:y + 2], np.array([222, 230, 244], np.uint8))
    cands = fold.line_candidates(img, DPI)[0]
    assert sum(c["score"] >= 0.8 for c in cands) >= 3                             # the rules are fold-like lines on their own ...
    assert fold.detect_folds(img, DPI) == []                                      # ... but periodic: crowded, so no fold
    on, _, st = pipeline.process_array(img.copy(), DPI, Options(), allow_drop=False, resize=False)
    off, _, _ = pipeline.process_array(img.copy(), DPI, Options(folds="off"), allow_drop=False, resize=False)
    assert st["folds"] == [] and np.array_equal(on, off)


def test_printed_table_not_a_fold():
    img = text_page(gap=(0, 0))
    x0, x1 = int(0.8 * DPI), int(7.7 * DPI)
    for f in (0.2, 0.34, 0.5, 0.66, 0.8):                                          # uniform mid-gray rules that stop inside the margins
        y = int(f * img.shape[0])
        img[y:y + 3, x0:x1] = 190
    for x in np.linspace(x0, x1, 5).astype(int):
        img[int(0.2 * img.shape[0]):int(0.8 * img.shape[0]), x:x + 3] = 190
    assert fold.detect_folds(img, DPI) == []
    on, _, st = pipeline.process_array(img.copy(), DPI, Options(), allow_drop=False, resize=False)
    off, _, _ = pipeline.process_array(img.copy(), DPI, Options(folds="off"), allow_drop=False, resize=False)
    assert st["folds"] == [] and np.array_equal(on, off)


def test_plain_page_without_crease_is_identical_to_off():
    img = text_page()
    on, _, st = pipeline.process_array(img.copy(), DPI, Options(), allow_drop=False, resize=False)
    off, _, _ = pipeline.process_array(img.copy(), DPI, Options(folds="off"), allow_drop=False, resize=False)
    assert st["folds"] == [] and np.array_equal(on, off)


def test_small_pages_are_skipped():
    small = crease(paper(4.0, 6.0))
    assert fold.line_candidates(small, DPI)[0] == [] and fold.detect_folds(small, DPI) == []
    assert len(fold.detect_folds(crease(paper(6.0, 8.0)), DPI)) == 1             # the same crease on a big enough sheet
    out, _, st = pipeline.process_array(small, DPI, Options(), allow_drop=False, resize=False)
    assert st["folds"] == []


def test_blank_page_with_only_a_crease_is_still_dropped():
    img = crease(paper())
    out, _, st = pipeline.process_array(img.copy(), DPI, Options(), allow_drop=True, resize=False)
    assert out is None and st["dropped"] and st["blank_score"] < 15
    assert len(st["folds"]) == 1                                                  # (the crease was found and removed before the score)
    out, _, st = pipeline.process_array(img.copy(), DPI, Options(), allow_drop=False, resize=False)
    assert out is not None and st["blank"] and band_px(out, level=250) < 100


def test_tier_b_needs_the_partner_and_tier_f_the_family():
    img = crease(paper(), y_frac=0.25)                                             # not a customary row: no tier A
    assert fold.detect_folds(img, DPI) == []
    partner = fold.line_candidates(crease(paper(seed=5), y_frac=0.252, seed=9), DPI)[0]
    got = fold.detect_folds(img, DPI, partner=partner)
    assert [f.info["tier"] for f in got] == ["B"]
    assert fold.detect_folds(img, DPI, partner=fold.line_candidates(paper(), DPI)[0]) == []
    c = lambda y, s: dict(y_row=int(y * 1000), y_frac=y, angle=0.0, amp=0.05, score=s,   # noqa: E731
                          stats=dict(pres=1.0, rel0=1.0, rel1=1.0, cv_mean=1.0, n_obs=30.0, step=0.0, cont=1.0))
    got = fold.decide([c(0.335, 1.2), c(0.67, 0.7), c(0.8, 0.7)], 8.5, 11.0)
    assert [(f.info["tier"], round(f.y_frac, 3)) for f in got] == [("A", 0.335), ("F", 0.67)]


def test_hint_from_stage_zero_gives_the_same_page(creased):
    hint = pipeline.FoldHint(own=fold.line_candidates(creased, DPI)[0], t=0.5)
    a, _, st_a = pipeline.process_array(creased.copy(), DPI, Options(deskew=False), allow_drop=False, resize=False, fold_hint=hint)
    b, _, st_b = pipeline.process_array(creased.copy(), DPI, Options(deskew=False), allow_drop=False, resize=False)
    assert np.array_equal(a, b) and st_a["folds"] == st_b["folds"] and st_a["t"]["fold_detect"] >= 0.5


def test_fold_failure_leaves_the_page_as_with_folds_off(creased, monkeypatch, caplog):
    def boom(*a, **k):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(fold, "detect_folds", boom)
    with caplog.at_level(logging.WARNING, logger="scanpp"):
        out, _, st = pipeline.process_array(creased.copy(), DPI, Options(), allow_drop=False, resize=False)
    off, _, _ = pipeline.process_array(creased.copy(), DPI, Options(folds="off"), allow_drop=False, resize=False)
    assert np.array_equal(out, off) and st["folds"] == []
    assert any("fold removal failed" in r.getMessage() and "synthetic failure" in r.getMessage() for r in caplog.records)


def test_removal_failure_does_not_touch_the_page(creased, monkeypatch, caplog):
    monkeypatch.setattr(fold, "fold_mask", lambda *a, **k: (_ for _ in ()).throw(ValueError("mask")))
    page = creased.copy()
    with caplog.at_level(logging.WARNING, logger="scanpp"):
        out, _, st = pipeline.process_array(page, DPI, Options(deskew=False, edge_clean=False), allow_drop=False, resize=False)
    assert st["folds"] == [] and any("fold removal failed" in r.getMessage() for r in caplog.records)
    assert np.array_equal(page, creased)


def test_folds_option_parses_and_propagates():
    ns, _ = cli.parse_args(["a.pdf"])
    assert ns.folds == "auto" and cli.options_from(ns).folds == "auto"
    ns, _ = cli.parse_args(["--folds", "off", "a.pdf"])
    assert cli.options_from(ns).folds == "off"
    ns, keep = cli.parse_args(["application/pdf", "false", "/x/a.pdf", "--folds off --keep-blank"])
    assert cli.options_from(ns).folds == "off" and keep == "false"
    with pytest.raises(SystemExit):
        cli.parse_args(["--folds", "maybe", "a.pdf"])
