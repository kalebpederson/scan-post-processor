"""Synthetic-image tests of the ported algorithms."""
import colorsys

import cv2
import numpy as np
import pytest

from helpers import DPI, blank_page, text_page
from scanpp.clean import clean_page
from scanpp.geometry import deskew, estimate_skew, is_blank, should_deskew, blank_score


def test_yellow_paper_flattens_to_white_text_stays_black():
    img = text_page(paper=(250, 235, 120), noise=2.0)
    out = clean_page(img, DPI)
    ink = (img.astype(int).sum(2) < 250)                      # darkest text pixels in the source
    paper = ~cv2.dilate(ink.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert (out[paper].min(axis=1) >= 250).mean() > 0.999       # paper -> white
    assert out[ink].mean() < 60                                 # ink stays dark


def test_keep_mode_keeps_tint():
    img = text_page(paper=(250, 235, 120))
    out = clean_page(img, DPI, paper="keep")
    med = np.median(out.reshape(-1, 3), axis=0)
    assert med[2] < med[0] - 60                                 # still yellow (B << R)


def test_skew_estimate_and_deskew_roundtrip():
    img = text_page(angle=2.0)                                 # content rotated CCW by 2 deg
    ang, conf = estimate_skew(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), DPI)
    assert abs(ang - 2.0) < 0.3, (ang, conf)
    assert should_deskew(ang, conf)
    fixed = deskew(img, ang, (255, 255, 255))
    ang2, _ = estimate_skew(cv2.cvtColor(fixed, cv2.COLOR_RGB2GRAY), DPI)
    assert abs(ang2) < 0.3
    ref = text_page(angle=0.0)
    # restored page correlates with the straight original much better than the skewed one
    err = lambda a: np.abs(a[150:-150, 150:-150].astype(int) - ref[150:-150, 150:-150]).mean()
    assert err(fixed) < err(img) * 0.7


def test_straight_page_not_rotated():
    ang, conf = estimate_skew(cv2.cvtColor(text_page(), cv2.COLOR_RGB2GRAY), DPI)
    assert not should_deskew(ang, conf)


def test_blank_with_dust_is_blank():
    # synthetic specks are crisp 1 px dots of 100 gray levels (~1 score point each; real dust is fainter): 10 of them
    # stay under the default threshold, 25 do not (they would only be dropped with a higher --blank-threshold)
    assert is_blank(blank_page(specks=10), DPI)
    assert not is_blank(blank_page(specks=25), DPI) and is_blank(blank_page(specks=25), DPI, threshold=40)


def test_pencil_word_not_blank():
    img = blank_page(paper=(245, 245, 243), specks=10)
    cv2.putText(img, "Pencil", (300, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (140, 140, 140), 3, cv2.LINE_AA)
    assert not is_blank(img, DPI), blank_score(img, DPI)


def test_text_page_not_blank():
    assert not is_blank(text_page(), DPI)


@pytest.mark.parametrize("paper,tol", [((252, 252, 250), 15), ((250, 235, 120), 35)])
def test_colored_ink_hue_preserved(paper, tol):
    # on tinted paper the flat-field division legitimately shifts hue a little (ink is seen through the paper tint)
    img = text_page(paper=paper, lines=0)
    strokes = {"red": (200, 30, 40), "blue": (30, 60, 200), "green": (20, 150, 60)}
    for i, col in enumerate(strokes.values()):
        cv2.line(img, (100, 200 + 150 * i), (700, 220 + 150 * i), col, 7, cv2.LINE_AA)
    out = clean_page(img, DPI)
    for i, col in enumerate(strokes.values()):
        y = 210 + 150 * i
        px = out[y - 1:y + 2, 380:420].reshape(-1, 3).mean(0)
        h_in = colorsys.rgb_to_hsv(*[c / 255 for c in col])[0]
        h_out, s_out, v_out = colorsys.rgb_to_hsv(*[c / 255 for c in px])
        dh = min(abs(h_in - h_out), 1 - abs(h_in - h_out)) * 360
        assert dh < tol, (col, px, dh)
        assert s_out > 0.4                                       # still clearly coloured
