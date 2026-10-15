"""Metric (b): stroke continuity of cleaned halftone regions vs a DESCREENED reference.
Reference strokes S: native flat-field ink density d (same as the cleaner's), Gaussian-blurred at ref_sigma_in (0.008 in = about
one halftone pitch / 2.4 px at 300 dpi) -> dref; S = pixels with dref >= max(0.04, 0.5 * p95(dref | dref > 0.03)) in the region,
excluding anything near dark ink (dref > 0.45 dilated 0.02 in) so only the LIGHT layer is scored.
Output density o = 1 - min_c(out_c)/255.
  (cov_25 / frag25 / dots25 use o >= 0.25, i.e. a clearly visible light-gray stroke; *_mid use o >= 0.45)
  cov_vis  = fraction of S with o >= 0.10   (non-white; R2's 'visible')
  cov_mid  = fraction of S with o >= 0.45   (at least mid-gray: stroke is drawn, not a faint ghost)
  n_ref    = connected components (8-conn) of S with area >= 0.0001 in^2
  n_out    = same for B = (o >= 0.45) restricted to dilate(S, 0.012 in)   (fragments of the stroke set)
  frag     = n_out / max(n_ref, 1)  (1 = same topology as the descreened reference; >1 = broken into pieces)
  dots     = fraction of B-components with area < 0.0006 in^2 (isolated dots)
"""
import sys, os
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2, numpy as np
from find_regions import density

def ell(dpi, r_in):
    r = max(1, int(round(r_in * dpi)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

def ref_strokes(rgb_crop, dpi, d=None, sigma_in=0.008):
    d = density(rgb_crop, dpi) if d is None else d
    dref = cv2.GaussianBlur(d, (0, 0), max(0.8, sigma_in * dpi))
    v = dref[dref > 0.03]
    if v.size < 20: return None, dref
    thr = max(0.04, 0.5 * float(np.percentile(v, 95)))
    S = dref >= thr
    dark = cv2.dilate((dref > 0.45).astype(np.uint8), ell(dpi, 0.02)) > 0
    return S & ~dark, dref

def ncc(m, min_px):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return int((st[1:, cv2.CC_STAT_AREA] >= min_px).sum()), st[1:, cv2.CC_STAT_AREA]

def score(S, out_crop, dpi):
    if S is None or S.sum() < 30: return None
    o = 1.0 - out_crop.min(-1).astype(np.float32) / 255.0
    a_ref = max(2, int(0.0001 * dpi * dpi))
    n_ref, _ = ncc(S, a_ref)
    near = cv2.dilate(S.astype(np.uint8), ell(dpi, 0.012)) > 0
    res = dict(cov_vis=float((o[S] >= 0.10).mean()), cov_25=float((o[S] >= 0.25).mean()), cov_mid=float((o[S] >= 0.45).mean()),
               n_ref=n_ref, tone=float(o[S].mean()))
    for tag, t in (("25", 0.25), ("mid", 0.45)):
        n_out, areas = ncc((o >= t) & near, a_ref)
        res["frag" + tag] = n_out / max(n_ref, 1)
        res["dots" + tag] = float((areas < 0.0006 * dpi * dpi).sum() / max(len(areas), 1))
    return res

def pad_box(box, dpi, shape, p_in=0.04):
    x, y, w, h = box; p = int(p_in * dpi)
    return max(0, x - p), max(0, y - p), min(shape[1], x + w + p), min(shape[0], y + h + p)
