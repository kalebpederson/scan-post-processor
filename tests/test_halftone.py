"""Halftoned light print: the cleaner must keep glyph strokes continuous and visible (hysteresis support + descreened tone)."""
import cv2
import numpy as np
import pytest

from halftone_synth import halftone, render_text
from scanpp.clean import clean_page

OFF = dict(desc_sigma_in=0)       # the previous cleaner (per-pixel gate, hysteresis on the denoised density only)


def _page(lpi, tone, scale, thickness, blur, dot_value=60):
    cov = render_text(scale=scale, thickness=thickness)
    g = halftone(cov * tone, lpi=lpi, blur_px=blur, dot_value=dot_value)
    return np.repeat(g[..., None], 3, 2), cov > 0.5


def _stats(out, ref, thr=0.25):
    d = 1.0 - out.min(-1) / 255.0
    n_cc = cv2.connectedComponents((d >= thr).astype(np.uint8), connectivity=8)[0] - 1
    return n_cc, float((d[ref] >= thr).mean())


@pytest.mark.parametrize("lpi,tone,scale,thickness,blur,max_cc_ratio,min_cov",
                         [(150, 0.30, 1.2, 2, 1.0, 1.3, 0.85),      # ~9 pt light gray at 150 lpi
                          (175, 0.25, 1.0, 2, 0.9, 2.6, 0.60)])     # lighter / finer screen: weaker but still coherent
def test_halftone_text_is_continuous(lpi, tone, scale, thickness, blur, max_cc_ratio, min_cov):
    rgb, ref = _page(lpi, tone, scale, thickness, blur)
    n_ref = cv2.connectedComponents(ref.astype(np.uint8), connectivity=8)[0] - 1      # components of the descreened glyphs
    n_new, cov_new = _stats(clean_page(rgb, 300), ref)
    n_old, cov_old = _stats(clean_page(rgb, 300, **OFF), ref)
    assert n_new <= max_cc_ratio * n_ref + 2, (n_new, n_ref)       # strokes are not broken into dots
    assert cov_new >= min_cov, cov_new                             # and are visible (>= 25 % dark) on the glyph area
    assert n_new < n_old or cov_new > cov_old + 0.1                # strictly better than the old gate (which dropped / fragmented them)
    assert cov_new >= cov_old - 0.02


def test_halftone_does_not_resurrect_bleed():
    """A blurred weak ghost of the same text (bleed-through) must still vanish."""
    cov = render_text(scale=1.2, thickness=2)
    ghost = cv2.GaussianBlur(cov, (0, 0), 3.0) * 0.10
    g = np.clip(255.0 * (1 - ghost) + np.random.default_rng(1).normal(0, 1.5, cov.shape), 0, 255).astype(np.uint8)
    out = clean_page(np.repeat(g[..., None], 3, 2), 300)
    d = 1.0 - out.min(-1) / 255.0
    assert (d > 0.05).mean() < 0.002
