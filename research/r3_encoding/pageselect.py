"""Recommended per-page encoder selection (round 2). rgb = cleaned 300 ppi page."""
import numpy as np, cv2
from encoders import *

def has_highlighter(rgb):
    mn = rgb.min(-1).astype(np.int16); mx = rgb.max(-1).astype(np.int16)
    pale = ((mx - mn) >= 25) & (mn >= 100) & (mn < 250)
    reg = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
    reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
    return reg.sum() >= 5000

def encode_page(rgb, info=None):
    mn = rgb.min(-1); ink = mn < 191; nonwhite = (rgb != 255).any(-1)
    if nonwhite.mean() < 2e-4: return Page(rgb.shape[1], rgb.shape[0]), "blank"
    sat = ((rgb.max(-1).astype(np.int16) - mn) > 40) & ink
    n_col = int(sat.sum()); gray = n_col < 1500 or n_col / max(1, ink.sum()) < 0.005
    if gray:   # no colour: mask + gray layer (no chroma issues)
        return enc_mrc(rgb, col_ppi=100, q=40, sub=0, thr=ADAPT, gray=True), "mrc300-c100-q40-gray"
    if has_highlighter(rgb) or ink.mean() > 0.15 or nonwhite.mean() > 0.30:     # natural look beats bytes on highlighter / ink-heavy art pages
        return enc_jpeg(rgb, r=150, q=55, sub=0), "jpeg150-q55-444"
    return enc_mrc(rgb, mask_ppi=300, codec="jbig2", col_ppi=100, q=40, sub=0, thr=ADAPT, colmode="mean"), "mrc300-c100-q40-444"
