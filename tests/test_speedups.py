"""The speed-optimised code paths against straightforward oracles (self-contained: no research/ data needed)."""
import cv2
import numpy as np
import pytest

from helpers import DPI, text_page
from scanpp import geometry as geo
from scanpp.clean import clean_page, clean_page_reference


def _imgs():
    rng = np.random.default_rng(7)
    yield "text", text_page(paper=(250, 240, 200), noise=2.0), DPI
    yield "text-deskewed", text_page(angle=1.5), DPI
    dark = np.full((400, 300, 3), 40, np.uint8)
    dark[100:260, 50:250] = 230
    yield "dark (dense gate)", dark, 150
    yield "noise", rng.integers(0, 255, (300, 220, 3), dtype=np.uint8), 100
    yield "white", np.full((250, 200, 3), 250, np.uint8), 100      # nothing passes the gate
    yield "tiny", rng.integers(0, 255, (1, 1, 3), dtype=np.uint8), 300
    yield "thin", rng.integers(0, 255, (200, 1, 3), dtype=np.uint8), 150


@pytest.mark.parametrize("paper", ["flatten", "keep"])
def test_clean_page_matches_reference(paper):
    for name, img, dpi in _imgs():
        a = clean_page(img, dpi, paper=paper)
        b = clean_page_reference(img, dpi, paper=paper)
        assert a.shape == b.shape and a.dtype == np.uint8, name
        assert np.abs(a.astype(int) - b).max() <= 1, name


def test_clean_page_gamma_uses_reference_path():
    img = text_page()
    assert np.array_equal(clean_page(img, DPI, gamma=1.2), clean_page_reference(img, DPI, gamma=1.2))


def test_clean_edges_equals_painting_the_full_mask():
    img = text_page(paper=(240, 240, 240))
    h, w = img.shape[:2]
    img = img.copy()
    img[:, :int(0.12 * DPI)] = 40                       # dark strip on the left edge
    img[-int(0.08 * DPI):, :] = 60                      # and the bottom
    for fill in (None, (255, 255, 255)):
        out, info = geo.clean_edges(img, DPI, fill)
        mask, _ = geo.edge_artifact_mask(img, DPI)
        ref = img.copy()
        ref[mask] = geo.paper_colour(img) if fill is None else fill
        assert mask.any()
        assert np.array_equal(out, ref)
        assert abs(info["painted_frac"] - mask.mean()) < 1e-12
    inplace = img.copy()
    res, _ = geo.clean_edges(inplace, DPI, (255, 255, 255), inplace=True)
    assert res is inplace and np.array_equal(res, out)


def test_paper_colour_matches_numpy_percentile():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (301, 177, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    c = img[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
    assert geo.paper_colour(img) == tuple(int(v) for v in np.percentile(c, 90, axis=0))


def test_projection_sparse_and_dense_paths_agree():
    """The sparse row-profile search and the rotate-the-image fallback (dense pages) find the same angle."""
    g = cv2.cvtColor(text_page(angle=-1.7), cv2.COLOR_RGB2GRAY)
    g, d = geo.prep_gray(g, DPI)
    ink = geo.ink_map(g, d)
    sparse = geo._skew_projection(g, d, 6.0, ink)
    dense_ink = _force_dense(ink)
    assert (dense_ink > 0.12).mean() > 0.15                  # takes the rotate-the-image fallback
    dense = geo._skew_projection(g, d, 6.0, dense_ink)
    assert abs(sparse - dense) < 0.15 and abs(sparse + 1.7) < 0.3


def _force_dense(ink):
    """Same ink pixels, plus enough weak-but-above-threshold speckle to exceed the 15 % density switch."""
    out = ink.copy()
    out[::3, ::2] = np.maximum(out[::3, ::2], 0.13)
    return out


# ---- helpers added with the neutral-snap / line-support speed-ups: each against its straightforward oracle
def test_comp_stats_equal_cv2_stats():
    from scanpp.clean import _comp_stats
    rng = np.random.default_rng(3)
    for thr in (0.5, 0.8, 0.95):
        m = (cv2.GaussianBlur(rng.random((150, 170)).astype(np.float32), (0, 0), 2) > np.quantile(rng.random(50), thr) * 0.55 + 0.3).astype(np.uint8)
        m[0, :] = 1                                             # components touching the border / spanning rows
        nl, lab, st, ce = cv2.connectedComponentsWithStats(m, connectivity=8)
        wd, ht, cent = _comp_stats(lab.ravel(), np.flatnonzero(m), m.shape[1], nl)
        assert np.array_equal(wd[1:], st[1:, cv2.CC_STAT_WIDTH]) and np.array_equal(ht[1:], st[1:, cv2.CC_STAT_HEIGHT])
        assert np.array_equal(cent[1:], ce[1:])


def test_open_runs_equal_morphological_opening():
    from scanpp.clean import _open_both
    rng = np.random.default_rng(1)
    for _ in range(400):
        h, w, L = int(rng.integers(40, 90)), int(rng.integers(40, 90)), int(rng.integers(3, 21))
        m = (rng.random((h, w)) < 0.03).astype(np.uint8)
        for _ in range(int(rng.integers(1, 12))):               # runs, some touching the borders
            if rng.random() < 0.5:
                y, x0 = rng.integers(h), rng.integers(-5, w)
                m[y, max(0, x0):max(0, min(w, x0 + rng.integers(1, 40)))] = 1
            else:
                x, y0 = rng.integers(w), rng.integers(-5, h)
                m[max(0, y0):max(0, min(h, y0 + rng.integers(1, 40))), x] = 1
        rh, rv = _open_both(np.flatnonzero(m), h, w, L)
        for ker, got in ((np.ones((1, L), np.uint8), rh), (np.ones((L, 1), np.uint8), rv)):
            g = np.zeros(h * w, np.uint8)
            g[got] = 1
            assert np.array_equal(cv2.morphologyEx(m, cv2.MORPH_OPEN, ker).ravel(), g)


@pytest.mark.parametrize("paper,dpi", [((255, 255, 255), 100), ((250, 235, 150), 150), ((190, 215, 245), 200)])
def test_neutral_sparse_equals_whole_plane(paper, dpi):
    from scanpp import clean as C
    img = text_page(paper=paper, noise=2.5, w=420, h=520)
    img[200:204, 20:400] = 0                                    # a black rule with a chromatic fringe band beside it
    img[204:206, 20:400] = np.clip(np.array(paper) * np.array([0.96, 1, 1]), 0, 255)
    img[100:400, 410:413] = 0                                   # and a vertical one
    img[100:400, 413:415] = np.clip(np.array(paper) * np.array([1, 1, 0.96]), 0, 255)
    img[300:318, 40:120] = (250, 200, 40)                       # a coloured patch
    h, w = img.shape[:2]
    bgs = np.maximum(C.estimate_background(img, dpi, 0.35, 1.5, 0.8, 50.0, 0.06), 16.0)
    q = [cv2.multiply(p, cv2.resize(np.ascontiguousarray(1.0 / bgs[..., c]), (w, h), interpolation=cv2.INTER_LINEAR), dtype=cv2.CV_32F)
         for c, p in enumerate(cv2.split(img))]
    qmin = np.minimum(cv2.min(cv2.min(q[0], q[1]), q[2]), 1.0)
    P = dict(C.DEFAULTS)
    pcol = C._paper_colour_bg(bgs)
    wn, we = C._neutral_weight(q, dpi, P, bgs, pcol)
    rng = np.random.default_rng(0)
    assert we is not None and we.sum() > 0                    # the fringe path is exercised
    for idx in (np.flatnonzero(qmin.ravel() < 0.9), np.sort(rng.choice(h * w, h * w // 5, replace=False))):
        a, b = C._neutral_sparse(q, qmin, idx, h, w, dpi, P, pcol)
        assert np.array_equal(a, wn.ravel()[idx])
        assert np.array_equal(np.zeros(idx.size, np.float32) if b is None else b, np.zeros(idx.size, np.float32) if we is None else we.ravel()[idx])


def test_clean_edges_g_returns_the_work_gray_only_when_unpainted():
    img = text_page(paper=(250, 250, 248), w=600, h=700)
    out, info, g, d = geo._clean_edges_g(img.copy(), DPI)
    assert info["painted_frac"] == 0 and g is not None
    g0, d0 = geo.prep_gray(img, DPI)
    assert d == d0 and np.array_equal(g, g0)
    strip = img.copy()
    strip[:, :30] = 20                                          # scanner strip -> painted
    out, info, g, d = geo._clean_edges_g(strip, DPI)
    assert info["painted_frac"] > 0 and g is None
