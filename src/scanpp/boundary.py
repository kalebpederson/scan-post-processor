"""Paper-boundary detection: the scanner lid / feeder backing beyond a short or narrow sheet.

Spiral-notebook pages and other sheets smaller than the scan area leave a band of non-paper (grey/beige backing, a black
shadow line, coloured cover fringes) along one or more page edges.  The strip/coil logic in `geometry` only reaches
0.35 in and only dark strips; this module finds the *paper edge* itself: a long, nearly border-parallel straight edge with
paper on the inner side and a non-paper region on the outer side, and reports the region beyond it so that it can be
painted with the paper colour (before page cleaning).

Safety first, because the region is *erased*.  A side is reported only when all of these hold:
  * the edge is straight and long: >= MIN_COVER of the side's length lies within a 0.03 in tolerance of one line, and
    the line is within MAX_TILT degrees of the border;
  * the paper side is paper (>= 90 % of a 0.25 in window), the outer side is *backing-like*: a run of non-paper that
    starts at the page border (>= 80 % of the rows between the border and the edge);
  * the region beyond is predominantly non-paper (a real contrast step, not just a rule on blank paper);
  * the region beyond carries no ink-like structure: strong gradients that are not long straight edges parallel to the
    border (text, strokes, graphics, coil teeth) are (almost) absent;
  * deep regions (> DEEP_IN) must additionally look like a backing: unsaturated and not black;
  * per column the painted depth never exceeds the measured edge position (a column whose paper starts earlier than
    the fitted line is only painted up to its own measured edge).
Everything runs at <= 100 ppi on the colour image; all lengths are in inches.  `find` returns the accepted boundaries,
`paint` applies them at full resolution.
"""
from __future__ import annotations

import cv2
import numpy as np

MIN_DEPTH_IN = 0.08        # an edge closer to the border than this is left to the strip logic
MAX_DEPTH_FRAC = 0.40      # an edge deeper than this fraction of the page (perpendicular extent) is not believed
MAX_DEPTH_IN = 4.0
MIN_COVER = 0.60           # fraction of the side's length that must lie on the fitted straight edge
MAX_TILT = 3.0             # deg
PAPER_TOL = 26             # max-channel difference from the paper colour above which a pixel is "non-paper"
EDGE_TOL_IN = 0.03         # inlier tolerance of the line fit
MIN_NONPAPER = 0.75        # fraction of the region beyond the edge that must differ from paper
MAX_RESID = 250            # ink-like strong-gradient pixels tolerated beyond the edge (100-ppi equivalents) ...
MAX_RESID_FRAC = 0.06      # ... and as a fraction of the region
DEEP_IN = 0.45             # beyond this depth the region must also look like a backing
DEEP_MAX_SAT = 0.35
DEEP_MAX_RESID = 120
DEEP_MIN_VAL = 0.30        # relative to the paper's brightness
SIDES = "TBLR"


def _orient(a: np.ndarray, side: str) -> np.ndarray:
    """View of `a` in which axis 0 is the depth from the border `side` and axis 1 runs along the border."""
    if side == "T":
        return a
    if side == "B":
        return a[::-1]
    if side == "L":
        return a.swapaxes(0, 1)
    return a[:, ::-1].swapaxes(0, 1)


def paper_colour(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    roi = rgb[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
    return np.percentile(roi[::7], 90, axis=0)


def _fit_line(xs: np.ndarray, ys: np.ndarray, tol: float, along: int):
    """Robust line through the points (RANSAC on random pairs, then least-squares on the inliers)."""
    n = len(xs)
    if n < 8:
        return None
    rng = np.random.default_rng(0)
    best = None
    for _ in range(80):
        i, j = rng.integers(0, n, 2)
        if abs(xs[i] - xs[j]) < max(20, 0.1 * along):
            continue
        s = (ys[j] - ys[i]) / (xs[j] - xs[i])
        if abs(np.degrees(np.arctan(s))) > MAX_TILT + 1:
            continue
        a = ys[i] - s * xs[i]
        inl = np.abs(ys - (a + s * xs)) <= tol
        if best is None or inl.sum() > best[2].sum():
            best = (a, s, inl)
    if best is None:
        return None
    a, s, inl = best
    if inl.sum() >= 8:
        s2, a2 = np.polyfit(xs[inl], ys[inl], 1)
        inl2 = np.abs(ys - (a2 + s2 * xs)) <= tol
        if inl2.sum() >= inl.sum():
            a, s, inl = float(a2), float(s2), inl2
    return float(a), float(s), inl


def detect_side(rgb: np.ndarray, dpi: float, side: str, paper: np.ndarray | None = None, along_excl: tuple | None = None):
    """Paper edge along `side` ('T','B','L','R') of the (<= 100 ppi) RGB image, or None.  The result carries 'depth' (edge
    depth in px of the oriented band for every position along the border) and the measurements used to accept it."""
    H, W = rgb.shape[:2]
    perp, along = (H, W) if side in "TB" else (W, H)
    dmax = int(min(MAX_DEPTH_FRAC * perp, MAX_DEPTH_IN * dpi))
    wp = max(4, int(round(0.25 * dpi)))
    if dmax < 0.4 * dpi or along < 2 * dpi:
        return None
    if paper is None:
        paper = paper_colour(rgb)
    pap = paper.astype(np.int16)
    full = _orient(rgb, side)
    # cheap pre-check: the rows around 0.05 in must be mostly non-paper (backing runs from the border to the edge)
    r0, r1 = max(1, int(0.03 * dpi)), max(3, int(0.07 * dpi) + 1)
    thin = np.abs(cv2.medianBlur(np.ascontiguousarray(full[:r1 + 2]), 5).astype(np.int16) - pap).max(axis=2)[r0:r1]
    if (thin > PAPER_TOL).mean() < 0.5:
        return None
    band = np.ascontiguousarray(full[:dmax + wp + 2])
    D = band.shape[0]
    dist = np.abs(cv2.medianBlur(band, 5).astype(np.int16) - pap).max(axis=2)
    N = (dist > PAPER_TOL).astype(np.float32)
    kx = max(3, int(round(0.4 * dpi))) | 1
    g = (cv2.blur(N, (kx, 1), borderType=cv2.BORDER_REPLICATE) >= 0.6).astype(np.int32)
    cs = np.concatenate([np.zeros((1, along), np.int32), np.cumsum(g, axis=0, dtype=np.int32)], axis=0)  # cs[d]=sum g[:d]
    ds = np.arange(1, D - wp)                         # candidate edge rows d: rows < d backing, rows d.. paper
    ok = ((cs[ds + wp] - cs[ds]) <= 0.1 * wp) & (cs[ds] >= 0.8 * ds[:, None]) & (g[ds - 1] == 1)
    has = ok.any(axis=0)
    b = np.where(has, ds[(len(ds) - 1) - np.argmax(ok[::-1], axis=0)], -1).astype(np.float64)
    cand = np.nonzero(has & (b >= MIN_DEPTH_IN * dpi))[0]
    if len(cand) < MIN_COVER * along * 0.8:
        return None
    tol = EDGE_TOL_IN * dpi + 0.5
    fit = _fit_line(cand.astype(np.float64), b[cand], tol, along)
    if fit is None:
        return None
    a, s, inl = fit
    cover = float(inl.sum() / along)
    tilt = float(np.degrees(np.arctan(s)))
    if cover < MIN_COVER or abs(tilt) > MAX_TILT:
        return None
    x = np.arange(along)
    line = a + s * x
    med = float(np.median(line))
    if med < MIN_DEPTH_IN * dpi or med > dmax:
        return None
    # painted depth: the line, but never deeper than a column's own measured edge where that is clearly shallower
    depth = np.where(has & (b < line - tol), b, line)
    depth = np.clip(depth, 0, dmax)
    res = dict(side=side, a=float(a), s=s, depth=depth, cover=cover, tilt=tilt, depth_in=med / dpi, dmax=dmax)
    # --- verification on the region beyond the edge
    rows = np.arange(D)[:, None]
    beyond = rows < (depth[None, :] - 1)
    if beyond.sum() < 50:
        return None
    res["nonpaper"] = float(N[beyond].mean())
    gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
    gb = cv2.GaussianBlur(gray, (0, 0), 0.7).astype(np.float32)
    gm = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    strong = (gm > 45).astype(np.uint8)
    lk = max(5, int(round(0.5 * dpi)))
    longpar = cv2.morphologyEx(cv2.dilate(strong, np.ones((3, 1), np.uint8)), cv2.MORPH_OPEN, np.ones((1, lk), np.uint8))
    resid = (strong > 0) & (cv2.dilate(longpar, np.ones((5, 1), np.uint8)) == 0)
    zone = beyond & (rows < depth[None, :] - 0.06 * dpi) & (rows > 0.03 * dpi)
    lo, hi = along_excl or (0, 0)
    ce = int(0.15 * dpi)                              # corners: dark page corners / bent paper are not content
    zone[:, :max(lo, ce)] = False
    zone[:, along - max(hi, ce):] = False
    nz = int(zone.sum())
    res["resid"] = float(resid[zone].sum()) if nz else 0.0
    res["resid_frac"] = float(resid[zone].mean()) if nz else 0.0
    if med > DEEP_IN * dpi and nz:
        # a deep backing is darker than the paper right at the edge (shadow line): per-column darkest row within 0.15 in beyond
        k0 = int(0.15 * dpi) + 1
        lum = gray.astype(np.float32)
        sh = np.array([lum[max(0, int(dd) - k0):max(1, int(dd) - 1), x].min() if dd > 2 else 255.0
                       for x, dd in enumerate(depth)])
        res["shadow"] = float(np.mean(sh < 0.6 * float(pap.mean())))
        hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
        z2 = zone & (rows < depth[None, :] - 0.2 * dpi)
        z2 = z2 if z2.sum() > 50 else zone
        res["sat"] = float(np.median(hsv[..., 1][z2])) / 255.0
        res["val"] = float(np.median(hsv[..., 2][z2])) / max(float(pap.max()), 1.0)
    return res


def accept(r: dict) -> bool:
    """The confidence rules (everything not already enforced in detect_side)."""
    if r["nonpaper"] < MIN_NONPAPER:
        return False
    if r["resid"] > MAX_RESID or r["resid_frac"] > MAX_RESID_FRAC:
        return False
    if "sat" in r:                                   # deep region: must look like a backing with a shadow line at the edge
        if r["sat"] > DEEP_MAX_SAT or r["val"] < DEEP_MIN_VAL or r["shadow"] < 0.5 or r["resid"] > DEEP_MAX_RESID:
            return False
    return True


def find(rgb: np.ndarray, dpi: float, skip: str = "", paper=None, coil: dict | None = None) -> list:
    """All confident paper boundaries of the (<= 100 ppi) RGB image, at most one per side.  Sides in `skip` are not
    examined (the coil logic owns them); `coil` maps 'L'/'R' to the coil depth in inches (the coil's teeth beyond a T/B
    edge are not counted as content)."""
    if paper is None:
        paper = paper_colour(rgb)
    out = []
    for side in SIDES:
        if side in skip:
            continue
        excl = None
        if coil and side in "TB":
            excl = (int(coil["L"] * dpi + 0.1 * dpi) if "L" in coil else 0, int(coil["R"] * dpi + 0.1 * dpi) if "R" in coil else 0)
        r = detect_side(rgb, dpi, side, paper, excl)
        if r is not None and accept(r):
            out.append(r)
    return out


def paint(img: np.ndarray, results: list, work_shape: tuple, fill) -> float:
    """Paint the region beyond every boundary of `results` (found on an image of `work_shape` = (h, w) pixels) into the
    full-resolution `img` in place.  Returns the painted fraction of the page (overlaps between sides counted twice)."""
    H, W = img.shape[:2]
    gh, gw = work_shape
    painted = 0
    for r in results:
        side = r["side"]
        view = _orient(img, side)                      # writable view
        along = view.shape[1]
        s_depth, s_along = ((H / gh, W / gw) if side in "TB" else (W / gw, H / gh))
        dw = r["depth"]
        xw = (np.arange(along) + 0.5) / s_along - 0.5
        depth = np.interp(xw, np.arange(len(dw)), dw) * s_depth             # px at full resolution, per column
        nb = int(min(np.ceil(depth.max()) + 1, view.shape[0]))
        m = np.arange(nb)[:, None] < depth[None, :]
        painted += int(m.sum())
        np.copyto(view[:nb], np.asarray(fill, img.dtype), where=m[..., None] if img.ndim == 3 else m)
    return painted / (H * W)


def region_mask(r: dict, shape: tuple) -> np.ndarray:
    """Boolean mask (H, W) of the area beyond boundary `r`, on an image of the analysis resolution `shape`."""
    m = np.zeros(shape, np.uint8)
    paint(m, [r], shape, 1)
    return m.astype(bool)
