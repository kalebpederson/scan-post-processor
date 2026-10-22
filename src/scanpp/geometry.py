"""Page geometry / page-level features for the scan post-processor.

Everything here is dpi-aware: inputs are processed at (at most) ~100 ppi internally and all lengths are
expressed in inches (converted with `dpi`).  See REPORT.md for the study behind every constant.

Public API
----------
estimate_skew(gray, dpi)            -> (angle_deg, confidence)       angle: CCW-positive skew of the content
deskew(rgb, angle, fill)            -> rgb                           rotates by -angle (cubic, same canvas size)
should_deskew(angle, conf)          -> bool                          confidence + minimum-angle rule
blank_score(img, dpi)               -> float                         0 = nothing there, larger = more ink
is_blank(img, dpi)                  -> bool                          blank_score < BLANK_THRESHOLD
edge_artifact_mask(img, dpi)        -> bool mask (full-res)          dark scanner strips + spiral-coil band
clean_edges(rgb, dpi, fill)         -> (rgb, info)                   paints the mask with paper colour

Angle convention: positive = content rotated counter-clockwise as seen on screen (a text baseline rising
towards the right has a positive angle).  deskew(img, estimate_skew(img)) therefore straightens the page.
"""
from __future__ import annotations
import numpy as np
import cv2

from . import boundary
from .options import DEFAULT_BLANK_THRESHOLD

# ----------------------------------------------------------------------------- constants (see REPORT.md)
WORK_DPI = 100            # internal analysis resolution (images are only ever down-scaled to this)
MAX_ANGLE = 5.0           # largest |skew| we search for / are willing to correct (deg)
MIN_ANGLE = 0.25          # do not rotate below this (deg)
MIN_CONF = 0.6            # >= 3 of the 4 independent estimators must agree (see estimate_skew)
VOTE_TOL = 0.3            # estimators "agree" when within this many degrees
BLANK_THRESHOLD = DEFAULT_BLANK_THRESHOLD   # blank_score below this -> blank (sharp-ink mass at 100 ppi; calibrated in README "Blank pages")


# ----------------------------------------------------------------------------- helpers
# (ported from research/r4_geometry/geometry.py; algorithms and defaults unchanged. `prep_gray` was `_prep`.)
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def prep_gray(gray: np.ndarray, dpi: float):
    """Down-scale to WORK_DPI if the input is finer than ~1.1x that.  Returns (gray, effective_dpi)."""
    g = _to_gray(gray)
    if dpi > WORK_DPI * 1.1:
        s = WORK_DPI / dpi
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        dpi = WORK_DPI
    return g, float(dpi)


def ink_map(gray: np.ndarray, dpi: float) -> np.ndarray:
    """Background-normalised ink density in [0,1] (0 = paper level).  Works on coloured paper because the
    local paper level (max filter over 0.25 in, median-cleaned, smoothed) is divided out.
    The smoothing (Gaussian sigma = window/3, the dominant cost) runs on a half-resolution copy and is interpolated
    back; the result differs from the full-resolution smoothing by < 0.004 in ink density on the corpus."""
    k = int(round(dpi * 0.25)) | 1
    h, w = gray.shape
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), 5).astype(np.float32)
    if min(h, w) >= 64:
        half = cv2.resize(bg, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.GaussianBlur(half, (0, 0), k / 6), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    np.maximum(bg, 1, out=bg)
    ink = cv2.divide(gray, bg, dtype=cv2.CV_32F)
    cv2.subtract(1.0, ink, ink)
    return np.clip(ink, 0, 1, out=ink)


def _paper_level(g: np.ndarray) -> float:
    h, w = g.shape
    return float(np.percentile(g[h // 4:3 * h // 4, w // 4:3 * w // 4], 90))


def _peak_hist(angles, weights, lo, hi, bin_=0.1, sm=0.5):
    """Peak (deg) of a gaussian-smoothed weighted histogram."""
    if len(angles) == 0:
        return 0.0, 0.0
    nb = int(round((hi - lo) / bin_))
    h, e = np.histogram(angles, bins=nb, range=(lo, hi), weights=weights)
    s = max(1, int(round(sm / bin_)))
    ker = np.exp(-0.5 * (np.arange(-3 * s, 3 * s + 1) / s) ** 2)
    ker /= ker.sum()
    hs = np.convolve(h, ker, mode="same")
    i = int(np.argmax(hs))
    pk = (e[i] + e[i + 1]) / 2
    m = np.abs(angles - pk) < 0.3
    return float(pk), float(weights[m].sum() / max(weights.sum(), 1e-9))


MIN_ANALYSIS_IN = 1.5     # analyses that need page structure (skew, coil/strip detection) need >= this on each side
MIN_ANALYSIS_PX = 64


def _big_enough(h: int, w: int, dpi: float) -> bool:
    """Is an h x w px region (at `dpi`) large enough for the structural analyses (>= 1.5 in and >= 64 px per side)?"""
    return min(h, w) >= max(MIN_ANALYSIS_PX, MIN_ANALYSIS_IN * dpi)


# ----------------------------------------------------------------------------- skew: four estimators
def _rot_gray(img, a):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), -a, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _row_var(im, a):
    """Variance of the row sums of `im` rotated by `a` degrees."""
    return float(cv2.reduce(_rot_gray(im, a), 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F).var())


def _parabolic_peak(angs, v):
    """Angle of the maximum of samples v(angs) (uniform grid), refined by a parabola through the 3 points around it."""
    j = int(np.argmax(v))
    if 0 < j < len(v) - 1:
        y0, y1, y2 = v[j - 1], v[j], v[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(angs[j] + 0.5 * (y0 - y2) / den * (angs[1] - angs[0]))
    return float(angs[j])


def _skew_projection(g, dpi, rng, ink=None):
    """(a) projection-profile variance maximisation, coarse-to-fine: 0.5 deg grid, then 0.2 deg (+-0.6) and 0.05 deg
    (+-0.1) grids, each refined by a parabola through the peak (was exhaustive 0.5 / 0.1 / 0.02 deg grids: 49 image
    rotations; same accuracy on the R4 ground truth).  The row profile of the rotated ink map is built by splatting
    the (sparse) ink pixels into row bins (linear in y) instead of warping the whole image; very dense pages fall back
    to rotating the image."""
    if ink is None:
        ink = ink_map(g, dpi)
    h, w = ink.shape
    ys, xs = np.nonzero(ink > 0.12)
    if len(ys) > 0.15 * h * w:
        ink = np.where(ink > 0.12, ink, 0).astype(np.float32)
        sm = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        coarse = lambda a: _row_var(sm, a)
        full = lambda a: _row_var(ink, a)
    else:
        v = ink[ys, xs].astype(np.float32)
        pad = int(np.ceil(np.sin(np.radians(rng + 1.0)) * w / 2)) + 2      # rows a point can leave the image by
        yc = ys.astype(np.float32) - np.float32(h / 2)
        xc = xs.astype(np.float32) - np.float32(w / 2)
        off = np.float32(h / 2 + pad)

        def profile(a, linear=True):
            r = np.radians(a)
            yy = yc * np.float32(np.cos(r))
            yy += xc * np.float32(np.sin(r))
            yy += off                                  # > 0 for every point: truncation == floor
            i = yy.astype(np.int32)
            n = h + 2 * pad + 2
            if not linear:
                return np.bincount(i, v, minlength=n)[pad:pad + h]
            yy -= i
            hist = np.bincount(i, v * (1 - yy), minlength=n)
            hist[1:] += np.bincount(i, v * yy, minlength=n)[:-1]       # weight f goes to row i + 1
            return hist[pad:pad + h]

        def coarse(a):
            return float(profile(a, False).var())

        def full(a):
            return float(profile(a).var())
    angs = np.arange(-rng, rng + 1e-9, 0.5)
    a0 = angs[np.argmax([coarse(a) for a in angs])]
    angs2 = np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.2)
    a1 = _parabolic_peak(angs2, [full(a) for a in angs2])
    angs3 = np.arange(a1 - 0.1, a1 + 0.1 + 1e-9, 0.05)
    return _parabolic_peak(angs3, [full(a) for a in angs3])


def _skew_lsd(g, dpi, rng, scale=0.75):
    """(b) line-segment detector (LSD, run at 0.75 x the 100 ppi work image: -25 % time, slightly better on the R4
    ground truth); length^2-weighted histogram of near-horizontal/vertical segments."""
    if scale != 1.0:
        g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        dpi = dpi * scale
    L = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(g)[0]
    if L is None:
        return 0.0
    L = L.reshape(-1, 4)
    dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
    ln = np.hypot(dx, dy)
    k = ln >= dpi * 0.3
    dx, dy, ln = dx[k], dy[k], ln[k]
    ang = (np.degrees(np.arctan2(-dy, dx)) + 90) % 180 - 90
    dev = np.where(np.abs(ang) <= 45, ang, ang - np.sign(ang) * 90)
    m = np.abs(dev) <= rng
    return _peak_hist(dev[m], ln[m] ** 2, -rng, rng, 0.1, 0.5)[0]


def _skew_cc(g, dpi, rng, ink=None):
    """(c) connected-component / text-line: word-level blobs (horizontal closing), minAreaRect long-side angle."""
    if ink is None:
        ink = ink_map(g, dpi)
    b = (ink > 0.25).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(dpi * 0.12)), 1)))
    cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    A, W = [], []
    for c in cs:
        if len(c) < 20:
            continue
        (_, _), (w, h), a = cv2.minAreaRect(c)
        if max(w, h) < dpi * 0.35 or max(w, h) / max(min(w, h), 1) < 2.5 or min(w, h) > dpi * 0.5:
            continue
        if w < h:
            a += 90
        a = (-a + 90) % 180 - 90
        A.append(a)
        W.append(max(w, h))
    A, W = np.array(A), np.array(W)
    m = np.abs(A) <= rng
    return _peak_hist(A[m], W[m], -rng, rng, 0.25, 0.6)[0]


def _skew_grad(g, dpi, rng):
    """(d) gradient-orientation histogram folded mod 90 deg (Scharr; strongest 3 % of edges)."""
    gf = cv2.GaussianBlur(g, (0, 0), 1.0).astype(np.float32)
    gx, gy = cv2.Scharr(gf, cv2.CV_32F, 1, 0), cv2.Scharr(gf, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx, gy)
    m = (mag > np.percentile(mag, 97)) & (mag > 60)
    d = (np.degrees(np.arctan2(gy[m], gx[m])) + 45) % 90 - 45
    dev = -d
    k = np.abs(dev) <= rng
    return _peak_hist(dev[k], mag[m][k], -rng, rng, 0.1, 0.4)[0]


def skew_estimates(gray: np.ndarray, dpi: float, max_angle: float = MAX_ANGLE) -> dict:
    """The four individual estimates (deg).  The outer 0.5 in (left/right) and 0.25 in (top/bottom) are
    cropped first: spiral coils and feeder strips there produce perfectly horizontal 'lines' that bias LSD/Hough."""
    g, d = prep_gray(gray, dpi)
    h, w = g.shape
    mx, my = int(round(0.5 * d)), int(round(0.25 * d))
    if not _big_enough(h - 2 * my, w - 2 * mx, d):
        return dict(proj=0.0, lsd=0.0, cc=0.0, grad=0.0)       # too small to measure a skew
    g = np.ascontiguousarray(g[my:h - my, mx:w - mx])
    rng = max_angle + 1.0
    ink = ink_map(g, d)                       # shared by the projection and blob estimators
    return dict(proj=_skew_projection(g, d, rng, ink), lsd=_skew_lsd(g, d, rng), cc=_skew_cc(g, d, rng, ink),
                grad=_skew_grad(g, d, rng))


_VOTE_ORDER = ("proj", "lsd", "cc", "grad")
_VOTE_W = np.array([3.0, 1.5, 1.0, 0.5])   # weights used when averaging the agreeing cluster (GT-validated)


def estimate_skew(gray: np.ndarray, dpi: int, max_angle: float = MAX_ANGLE):
    """Estimate page skew.

    Returns (angle_deg, confidence).  Four independent estimators (projection profile, LSD, word-blob minAreaRect,
    gradient orientation) vote: the largest cluster of estimates within VOTE_TOL degrees wins, angle = its weighted mean
    (weights 3/1.5/1/0.5 in that order).
    confidence = (cluster_size - 1) / 3 in {0, 1/3, 2/3, 1} (4/4 agree -> 1.0, 3/4 -> 0.67, 2/4 -> 0.33, none -> 0)
    and is 0 if |angle| > max_angle.  Use confidence >= MIN_CONF (3 of 4 agree) as "reliable".
    Blank / nearly blank pages have no structure to measure; the caller should skip them (angle is meaningless).
    """
    est = skew_estimates(gray, dpi, max_angle)
    if not any(est.values()):
        return 0.0, 0.0
    e = np.array([est[k] for k in _VOTE_ORDER])
    best = None
    for j in range(len(e)):
        s = np.abs(e - e[j]) < VOTE_TOL
        if best is None or s.sum() > best.sum():
            best = s
    w = _VOTE_W * best
    ang = float((e * w).sum() / w.sum())
    conf = (int(best.sum()) - 1) / 3.0
    if abs(ang) > max_angle:
        conf = 0.0
    return ang, conf


def should_deskew(angle: float, conf: float, min_angle: float = MIN_ANGLE, min_conf: float = MIN_CONF,
                  max_angle: float = MAX_ANGLE) -> bool:
    """Rotate only when the estimate is reliable and large enough to matter (see REPORT.md for justification)."""
    return conf >= min_conf and min_angle <= abs(angle) <= max_angle


def _pct90_u8(img: np.ndarray, c: int = 0) -> int:
    """int(np.percentile(channel c of img, 90)) for a uint8 image (view), from a 256-bin histogram (no sort / copy)."""
    hist = cv2.calcHist([img], [c], None, [256], [0, 256]).ravel().astype(np.int64)
    cs = np.cumsum(hist)
    n = int(cs[-1])
    pos = 0.9 * (n - 1)
    lo = int(np.floor(pos))
    a = int(np.searchsorted(cs, lo + 1))            # value of the lo-th order statistic
    b = int(np.searchsorted(cs, lo + 2)) if lo + 1 < n else a
    return int(a + (b - a) * (pos - lo))


def paper_colour(rgb: np.ndarray) -> tuple:
    """Estimate the paper colour: per-channel 90th percentile of the central half of the page."""
    h, w = rgb.shape[:2]
    roi = rgb[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return tuple(_pct90_u8(roi, c) for c in range(3))


def deskew(rgb: np.ndarray, angle: float, fill=None) -> np.ndarray:
    """Rotate `rgb` by -angle (i.e. undo a measured skew of `angle`) about the page centre, keeping the canvas size.
    Uses INTER_CUBIC (best SSIM/speed trade-off, see REPORT.md).  `fill` (colour tuple/int) paints the exposed
    corners; default = estimated paper colour (pass (255,255,255) after cleaning)."""
    if fill is None:
        fill = paper_colour(rgb) if rgb.ndim == 3 else int(np.percentile(rgb, 90))
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


# ----------------------------------------------------------------------------- edge artefacts
def _edge_runs(g, side, dark_thr, maxd):
    """Length (px) of the contiguous dark run starting at the page edge, for every position along `side`."""
    a = {"L": g[:, :maxd], "R": g[:, ::-1][:, :maxd], "T": g[:maxd, :].T, "B": g[::-1, :][:maxd, :].T}[side]
    dark = a < dark_thr
    run = np.argmin(dark, axis=1)
    return np.where(dark.all(1), maxd, run)


def _coil_extent(run, dpi):
    """Spiral-binding coil: the dark-run profile along the edge is strongly periodic (pitch ~0.25 in = 24.8 px @100 ppi)
    and several teeth deep.  Returns coil depth in px (95th percentile of the run = tooth tips), 0 if no coil."""
    n = len(run)
    k = int(round(0.8 * dpi)) | 1
    if n < k + max(32, int(dpi)):                       # too short for the detrending kernel + a periodicity estimate
        return 0
    x = run.astype(float) - np.convolve(run, np.ones(k) / k, mode="same")      # detrend: kills smooth wedges
    x = x[k // 2:n - k // 2]
    if len(x) < max(32, int(dpi)) or x.std() < 1e-6 or np.mean(run >= 0.03 * dpi) < 0.3:
        return 0
    F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x)) * dpi                  # cycles per inch
    band = (fr > 2.2) & (fr < 11.0)                     # pitch 0.09 - 0.45 in
    if F[1:].sum() <= 0 or F[band].max() / F[1:].sum() < 0.3:
        return 0
    return int(np.percentile(run, 95))


def _edge_mask_work(img: np.ndarray, dpi: float, max_strip_in: float = 0.35):
    """edge_artifact_mask at the working resolution (no upsampling).  Returns (mask, info, maxd)."""
    return _edge_mask_work_g(*prep_gray(img, dpi), max_strip_in)


def edge_artifact_mask(img: np.ndarray, dpi: float, max_strip_in: float = 0.35):
    """Full-resolution boolean mask of non-paper edge artefacts:
      * dark/grey scanner-background strips and wedges hugging a page edge (contiguous dark run from the edge,
        depth <= max_strip_in, grown by 0.02 in; rows/cols whose run exceeds max_strip_in are left alone), and
      * the spiral-binding coil on the left/right edge (periodic dark teeth, ~0.3-0.4 in deep): whole band masked, and
      * the whole area beyond a confidently detected straight paper edge (scanner lid / backing behind a short or narrow
        sheet; any depth up to 40 % of the page, see `boundary`).
    Returns (mask, info)."""
    g, d = prep_gray(img, dpi)
    mask, info, maxd = _edge_mask_work_g(g, d, max_strip_in)
    if img.ndim == 3 and any(k.startswith("coil_") for k in info):
        n = _coil_fringe(mask, g, _work_rgb(img, g.shape), d, info, maxd)
        if n:
            info["fringe_px"] = n
    for r in _paper_boundaries(img, g, d, info):         # deep regions beyond a confidently detected paper edge
        mask |= boundary.region_mask(r, mask.shape)
        info["boundary_" + r["side"]] = r["depth_in"]
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, info


def _edge_mask_work_g(g, d, max_strip_in=0.35):
    h, w = g.shape
    mask = np.zeros((h, w), bool)
    info = {}
    if not _big_enough(h, w, d):                        # tiny image: no scanner edge to speak of
        return mask, info, 0
    pl = _paper_level(g)
    pad = max(1, int(round(0.02 * d)))
    maxd = int(round(0.6 * d))
    cap = int(round(max_strip_in * d))
    for s in "LRTB":
        run = _edge_runs(g, s, 0.75 * pl, maxd)
        coil = _coil_extent(run, d) if s in "LR" else 0
        ok = (run >= 1) & (run <= cap)
        depth = np.where(ok, run + pad, 0)
        if coil:
            cd = min(coil + int(round(0.06 * d)), maxd)
            depth = np.maximum(depth, cd)
            info["coil_" + s] = coil / d
        info["strip_" + s] = float(np.percentile(np.where(ok, run, 0), 99)) / d
        n = len(depth)
        if s in "LR":
            m = np.arange(maxd)[None, :] < depth[:, None]
            if s == "L":
                mask[:, :maxd] |= m
            else:
                mask[:, w - maxd:] |= m[:, ::-1]
        else:
            m = np.arange(maxd)[None, :] < depth[:, None]
            if s == "T":
                mask[:maxd, :] |= m.T
            else:
                mask[h - maxd:, :] |= m.T[::-1, :]
    if any(k.startswith("coil_") for k in info):         # spiral notebook on the glass: the lid frame's penumbra
        for s in "TBLR":
            if "coil_" + s not in info:
                _border_hairline(mask, g, s, pl, d, info)
    n = _corner_dots(mask, g, pl, d)
    if n:
        info["corner_dots"] = n
    return mask, info, maxd


# research switches (r7): False reproduces the previous behaviour exactly (see research/r7_flatbed)
FLATBED_CORNER_DOTS = True
FLATBED_HAIRLINE = True
HAIRLINE_FRAC = 0.7       # share of the border (corners excluded) that must be shaded
HAIRLINE_MAX_IN = 0.04
HAIRLINE_DROP = 12        # levels below paper for "shaded"


def _border_hairline(mask, g, side, pl, d, info) -> None:
    """Flatbed scan of a spiral notebook: the lid frame's penumbra leaves a light grey hairline (<= 0.04 in) along a whole
    border that is too light for the dark-strip run logic (< 0.75 paper).  Painted (only on pages with a coil, on borders
    without one) when >= 70 % of the border is shaded > HAIRLINE_DROP levels below paper or already painted, and at least
    30 % is shaded -- content never lies along the whole border."""
    if not FLATBED_HAIRLINE:
        return
    go = boundary._orient(g, side)
    mo = boundary._orient(mask, side)
    nb = max(1, int(round(HAIRLINE_MAX_IN * d)))
    c = int(0.15 * d)
    k = 0
    for r in range(nb):
        row = go[r, c:go.shape[1] - c]
        shaded = row < pl - HAIRLINE_DROP
        if row.size < 8 or float((shaded | mo[r, c:go.shape[1] - c]).mean()) < HAIRLINE_FRAC or float(shaded.mean()) < 0.3:
            break
        k += 1
    if k:
        mo[:k + 1] |= (go[:k + 1] < pl - HAIRLINE_DROP)
        info["hairline_" + side] = k
FLATBED_COIL_FRINGE = True


def _corner_dots(mask: np.ndarray, g: np.ndarray, pl: float, d: float) -> int:
    """Dark specks left in a page corner where two painted scanner strips meet (the strip runs of both borders are
    "too long" there, so neither is painted).  A connected dark blob (< 0.6 paper) is painted when it touches the image
    corner region, lies entirely within 0.15 in of both borders, and touches the already painted strip mask.  Returns the
    number of blobs painted."""
    if not FLATBED_CORNER_DOTS or not mask.any():
        return 0
    k = int(round(0.2 * d))
    lim = int(round(0.15 * d))
    h, w = g.shape
    if min(h, w) < 4 * k:
        return 0
    painted = 0
    for ys, xs in ((slice(0, k), slice(0, k)), (slice(0, k), slice(w - k, w)),
                   (slice(h - k, h), slice(0, k)), (slice(h - k, h), slice(w - k, w))):
        win_g, win_m = g[ys, xs], mask[ys, xs]
        dark = (win_g < 0.6 * pl) & ~win_m
        if not dark.any():
            continue
        near = cv2.dilate(win_m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        nl, lab = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
        # orient so that (0, 0) is the image corner
        oy = slice(None) if ys.start == 0 else slice(None, None, -1)
        ox = slice(None) if xs.start == 0 else slice(None, None, -1)
        lab_o = lab[oy, ox]
        for i in range(1, nl):
            yy, xx = np.nonzero(lab_o == i)
            if yy.min() != 0 and xx.min() != 0:                       # must touch an image border
                continue
            if yy.max() >= lim or xx.max() >= lim:                    # compact: within 0.15 in of both borders
                continue
            if not near[lab == i].any():
                continue
            blob = cv2.dilate((lab == i).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            win_m |= blob
            painted += 1
    return painted


def _coil_fringe(mask: np.ndarray, g: np.ndarray, rgb_w: np.ndarray, d: float, info: dict, maxd: int) -> int:
    """Grow the coil band over its *end* fringe (in place on `mask`; returns the added pixel count).  Where the spiral ends
    at the top/bottom of the page the paper lifts: a smooth grey/lilac shadow patch continues beyond the band and the
    cover shows as a thin bluish line along the border.  Within 0.9 in of the page ends and 0.25 in beyond the band (0.55 in for the border line), the
    pixels connected to the band that are (a) light shading (0.62 paper < gray < paper - 8) and unsaturated, forming
    blobs >= 0.05 in thick (strokes of pen/pencil are thinner or darker), or (b) a >= 0.15 in long run of coloured (saturated) pixels
    within 0.07 in of the page's top/bottom border, are added.  Handwriting is never connected through such blobs."""
    if not FLATBED_COIL_FRINGE:
        return 0
    pl = _paper_level(g)
    n = 0
    hsv = None
    k5 = max(3, int(round(0.05 * d))) | 1
    for side in "LR":
        if "coil_" + side not in info:
            continue
        if hsv is None:
            hsv = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2HSV)[..., 1]
        m_o, g_o, s_o = boundary._orient(mask, side), boundary._orient(g, side), boundary._orient(hsv, side)
        depth, along = m_o.shape
        cd = min(maxd, int(round(info["coil_" + side] * d + 0.06 * d)))
        zd = min(maxd, cd + int(round(0.25 * d)))
        ze = int(round(0.9 * d))
        zone = np.zeros(m_o.shape, bool)
        zone[:zd, :ze] = True
        zone[:zd, along - ze:] = True
        f = (g_o < pl - 8) & (g_o > 0.62 * pl) & (s_o < 0.15 * 255) & zone
        f = cv2.morphologyEx(f.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k5, k5))) > 0
        bw = max(1, int(round(0.07 * d)))
        edge = np.zeros(m_o.shape, bool)
        ze_d = cd + int(round(0.55 * d))                 # the cover line along the border runs further than the shadow
        edge[:ze_d, :bw] = True
        edge[:ze_d, along - bw:] = True
        e = (g_o < pl - 12) & (g_o > 0.55 * pl) & (s_o > 0.2 * 255) & edge
        e = cv2.morphologyEx(e.astype(np.uint8), cv2.MORPH_OPEN, np.ones((max(3, int(round(0.15 * d))), 1), np.uint8)) > 0
        cand = (f | e) & ~m_o
        grow = m_o.copy()
        cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        for _ in range(int(round(0.55 * d))):
            nxt = (cv2.dilate(grow.astype(np.uint8), cross) > 0) & (cand | grow)
            if (nxt == grow).all():
                break
            grow = nxt
        add = grow & ~m_o
        if add.any():
            add = cv2.dilate(add.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            add &= zone | edge
            n += int((add & ~m_o).sum())
            m_o |= add
    return n


def _work_rgb(rgb: np.ndarray, shape: tuple) -> np.ndarray:
    """The colour image at the working resolution of a gray image of `shape` (the image itself when no scaling is needed)."""
    if rgb.shape[:2] == tuple(shape):
        return rgb
    return cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)


def _paper_boundaries(rgb: np.ndarray, g: np.ndarray, d: float, info: dict) -> list:
    """Confident paper boundaries (see `boundary`) on every side the coil logic does not own; [] for tiny images."""
    if rgb.ndim != 3 or not _big_enough(*g.shape, d):
        return []
    skip = "".join(k[-1] for k in info if k.startswith("coil_"))
    coil = {k[-1]: v for k, v in info.items() if k.startswith("coil_")}
    return boundary.find(_work_rgb(rgb, g.shape), d, skip, None, coil)


def clean_edges(rgb: np.ndarray, dpi: float, fill=None, inplace: bool = False):
    """Paint edge artefacts with `fill` (default estimated paper colour; pass white after page cleaning):
      * scanner strips and the spiral coil (`edge_artifact_mask`, <= 0.6 in from the border), and
      * the whole area beyond a confidently detected straight paper edge (scanner lid / backing behind a short or
        narrow sheet, see `boundary`; any depth up to 40 % of the page).
    Returns (rgb, info) where info has the artefact depths in inches, `boundary_<side>` (edge depth in inches) for every
    painted paper boundary, and the fraction of pixels painted.  `inplace=True` paints into `rgb` itself (no copy).

    The strip mask is computed at the working resolution and only the (<= 0.6 in) edge bands are expanded to full
    resolution, with exactly the nearest-neighbour mapping of cv2.resize, so the result equals painting the full mask."""
    return _clean_edges_g(rgb, dpi, fill, inplace)[:2]


def _clean_edges_g(rgb: np.ndarray, dpi: float, fill=None, inplace: bool = False):
    """`clean_edges` -> (rgb, info, g, d): `g, d` is the working-resolution gray image of the page and its ppi (= prep_gray(rgb, dpi)) when no
    pixel was painted, else g is None (the caller needs the gray of the painted page then)."""
    g, d = prep_gray(rgb, dpi)
    mask, info, maxd = _edge_mask_work_g(g, d)
    if rgb.ndim == 3 and any(k.startswith("coil_") for k in info):
        info["fringe_px"] = _coil_fringe(mask, g, _work_rgb(rgb, g.shape), d, info, maxd)
    bounds = _paper_boundaries(rgb, g, d, info)            # detected on the unpainted image
    H, W = rgb.shape[:2]
    gh, gw = mask.shape
    if not mask.any() and not bounds:
        info["painted_frac"] = 0.0
        return (rgb if inplace else rgb.copy()), info, g, d
    out = rgb if inplace else rgb.copy()
    if fill is None:
        fill = paper_colour(rgb) if rgb.ndim == 3 else int(np.percentile(rgb, 90))
    xs = np.minimum(np.floor(np.arange(W) * (gw / W)).astype(np.int64), gw - 1)         # cv2.resize INTER_NEAREST mapping
    ys = np.minimum(np.floor(np.arange(H) * (gh / H)).astype(np.int64), gh - 1)
    comb = mask
    for r in bounds:
        comb = comb | boundary.region_mask(r, mask.shape)
        info["boundary_" + r["side"]] = r["depth_in"]
    cx, cy = np.bincount(xs, minlength=gw), np.bincount(ys, minlength=gh)
    painted = float(cy @ (comb.astype(np.int64) @ cx)) / (H * W)
    if mask.any():
        xl, xr = int(np.searchsorted(xs, maxd, "left")), int(np.searchsorted(xs, gw - maxd, "left"))
        yt, yb = int(np.searchsorted(ys, maxd, "left")), int(np.searchsorted(ys, gh - maxd, "left"))
        for sl_y, sl_x in ((slice(None), slice(0, xl)), (slice(None), slice(xr, W)),
                           (slice(0, yt), slice(None)), (slice(yb, H), slice(None))):
            band = mask[ys[sl_y]][:, xs[sl_x]]
            if band.any():
                np.copyto(out[sl_y, sl_x], np.asarray(fill, out.dtype), where=band[..., None] if out.ndim == 3 else band)
    if bounds:
        boundary.paint(out, bounds, (gh, gw), fill)
    info["painted_frac"] = painted
    return out, info, None, d


# ----------------------------------------------------------------------------- blank detection
SHARP_GRAD = 60.0         # Sobel(sigma 0.7) gradient magnitude (8-bit gray levels) that counts as a sharp edge
INK_T = 0.12              # ink-map threshold


def blank_map(img: np.ndarray, dpi: float, T: float = INK_T, grad_thr: float = SHARP_GRAD):
    """Per-pixel 'sharp ink' map at the working resolution: background-normalised ink where (a) ink > T, (b) the
    local gradient is edge-like (kills blurry bleed-through and smudges) and (c) the pixel is outside edge artefacts
    (feeder strips, spiral coil).  Returns (map, scale) with scale converting pixel counts to 100-ppi equivalents."""
    g, d = prep_gray(img, dpi)
    emask, _, _ = _edge_mask_work_g(g, d)
    ink = cv2.GaussianBlur(ink_map(g, d), (0, 0), 0.6 * d / 100)
    gb = cv2.GaussianBlur(g, (0, 0), 0.7 * d / 100).astype(np.float32)
    gm = cv2.magnitude(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    ok = cv2.bitwise_and(cv2.compare(ink, T, cv2.CMP_GT), cv2.compare(gm, grad_thr * d / 100, cv2.CMP_GT))
    if emask.any():
        ok[cv2.dilate(emask.view(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0
    return cv2.bitwise_and(ink, ink, mask=ok), (100.0 / d) ** 2


def blank_score(img: np.ndarray, dpi: float) -> float:
    """Sharp-ink mass of the page: sum of background-normalised ink over pixels that are darker than the paper by
    >12 %, have edge-like gradients, and are not scanner-edge strips / binding coil; in pixel-equivalents at 100 ppi.
    ~0 for blank sheets (dust and bleed-through give single digits, at most ~25 in the corpus), >=230 for every real labelled
    page; a short pencil line / lone page number / single light word land in between (see README, 'Blank pages')."""
    m, sc = blank_map(img, dpi)
    return float(m.sum() * sc)


def is_blank(img: np.ndarray, dpi: float, threshold: float = BLANK_THRESHOLD) -> bool:
    """True when the page carries no intentional mark (dust, bleed-through, edge shadows only)."""
    return blank_score(img, dpi) < threshold
