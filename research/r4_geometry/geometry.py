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

# ----------------------------------------------------------------------------- constants (see REPORT.md)
WORK_DPI = 100            # internal analysis resolution (images are only ever down-scaled to this)
MAX_ANGLE = 5.0           # largest |skew| we search for / are willing to correct (deg)
MIN_ANGLE = 0.25          # do not rotate below this (deg)
MIN_CONF = 0.6            # >= 3 of the 4 independent estimators must agree (see estimate_skew)
VOTE_TOL = 0.3            # estimators "agree" when within this many degrees
BLANK_THRESHOLD = 40.0    # blank_score below this -> blank (sharp-ink mass at 100 ppi; see REPORT.md)


# ----------------------------------------------------------------------------- helpers
def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def _prep(gray: np.ndarray, dpi: float):
    """Down-scale to WORK_DPI if the input is finer than ~1.1x that.  Returns (gray, effective_dpi)."""
    g = _to_gray(gray)
    if dpi > WORK_DPI * 1.1:
        s = WORK_DPI / dpi
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        dpi = WORK_DPI
    return g, float(dpi)


def ink_map(gray: np.ndarray, dpi: float) -> np.ndarray:
    """Background-normalised ink density in [0,1] (0 = paper level).  Works on coloured paper because the
    local paper level (max filter over 0.25 in, median-cleaned, smoothed) is divided out."""
    g = gray.astype(np.float32)
    k = int(round(dpi * 0.25)) | 1
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), 5).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    return np.clip(1 - g / np.maximum(bg, 1), 0, 1)


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


# ----------------------------------------------------------------------------- skew: four estimators
def _rot_gray(img, a):
    h, w = img.shape[:2]
    return cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), -a, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderValue=0)


def _skew_projection(g, dpi, rng):
    """(a) projection-profile variance maximisation, coarse-to-fine (0.5 deg @ half-res, 0.1, 0.02 deg)."""
    ink = ink_map(g, dpi)
    ink = np.where(ink > 0.12, ink, 0).astype(np.float32)
    var = lambda im, a: float(np.var(_rot_gray(im, a).sum(1)))
    sm = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    angs = np.arange(-rng, rng + 1e-9, 0.5)
    sc = np.array([var(sm, a) for a in angs])
    a0 = angs[sc.argmax()]
    angs2 = np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.1)
    a1 = angs2[np.argmax([var(ink, a) for a in angs2])]
    angs3 = np.arange(a1 - 0.1, a1 + 0.1 + 1e-9, 0.02)
    best = angs3[np.argmax([var(ink, a) for a in angs3])]
    return float(best)


def _skew_lsd(g, dpi, rng):
    """(b) line-segment detector (LSD); length^2-weighted histogram of near-horizontal/vertical segments."""
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


def _skew_cc(g, dpi, rng):
    """(c) connected-component / text-line: word-level blobs (horizontal closing), minAreaRect long-side angle."""
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
    mag = np.hypot(gx, gy)
    m = (mag > np.percentile(mag, 97)) & (mag > 60)
    d = (np.degrees(np.arctan2(gy[m], gx[m])) + 45) % 90 - 45
    dev = -d
    k = np.abs(dev) <= rng
    return _peak_hist(dev[k], mag[m][k], -rng, rng, 0.1, 0.4)[0]


def skew_estimates(gray: np.ndarray, dpi: float, max_angle: float = MAX_ANGLE) -> dict:
    """The four individual estimates (deg).  The outer 0.5 in (left/right) and 0.25 in (top/bottom) are
    cropped first: spiral coils and feeder strips there produce perfectly horizontal 'lines' that bias LSD/Hough."""
    g, d = _prep(gray, dpi)
    h, w = g.shape
    mx, my = int(round(0.5 * d)), int(round(0.25 * d))
    g = np.ascontiguousarray(g[my:h - my, mx:w - mx])
    rng = max_angle + 1.0
    return dict(proj=_skew_projection(g, d, rng), lsd=_skew_lsd(g, d, rng), cc=_skew_cc(g, d, rng), grad=_skew_grad(g, d, rng))


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


def paper_colour(rgb: np.ndarray) -> tuple:
    """Estimate the paper colour: per-channel 90th percentile of the central half of the page."""
    h, w = rgb.shape[:2]
    c = rgb[h // 4:3 * h // 4, w // 4:3 * w // 4].reshape(-1, 3)
    return tuple(int(v) for v in np.percentile(c, 90, axis=0))


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
    x = run.astype(float) - np.convolve(run, np.ones(k) / k, mode="same")      # detrend: kills smooth wedges
    x = x[k // 2:n - k // 2]
    if len(x) < 100 or x.std() < 1e-6 or np.mean(run >= 0.03 * dpi) < 0.3:
        return 0
    F = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    fr = np.fft.rfftfreq(len(x)) * dpi                  # cycles per inch
    band = (fr > 2.2) & (fr < 11.0)                     # pitch 0.09 - 0.45 in
    if F[1:].sum() <= 0 or F[band].max() / F[1:].sum() < 0.3:
        return 0
    return int(np.percentile(run, 95))


def edge_artifact_mask(img: np.ndarray, dpi: float, max_strip_in: float = 0.35):
    """Full-resolution boolean mask of non-paper edge artefacts:
      * dark/grey scanner-background strips and wedges hugging a page edge (contiguous dark run from the edge,
        depth <= max_strip_in, grown by 0.02 in; rows/cols whose run exceeds max_strip_in are left alone), and
      * the spiral-binding coil on the left/right edge (periodic dark teeth, ~0.3-0.4 in deep): whole band masked.
    Returns (mask, info)."""
    g, d = _prep(img, dpi)
    h, w = g.shape
    pl = _paper_level(g)
    mask = np.zeros((h, w), bool)
    info = {}
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
    if img.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask, info


def clean_edges(rgb: np.ndarray, dpi: float, fill=None):
    """Paint edge artefacts (scanner strips, spiral coil) with `fill` (default estimated paper colour; pass white after
    page cleaning).  Returns (rgb, info) where info has the artefact depths in inches and the fraction of pixels painted."""
    mask, info = edge_artifact_mask(rgb, dpi)
    if fill is None:
        fill = paper_colour(rgb) if rgb.ndim == 3 else int(np.percentile(rgb, 90))
    out = rgb.copy()
    out[mask] = fill
    info["painted_frac"] = float(mask.mean())
    return out, info


# ----------------------------------------------------------------------------- blank detection
SHARP_GRAD = 60.0         # Sobel(sigma 0.7) gradient magnitude (8-bit gray levels) that counts as a sharp edge
INK_T = 0.12              # ink-map threshold


def blank_map(img: np.ndarray, dpi: float, T: float = INK_T, grad_thr: float = SHARP_GRAD):
    """Per-pixel 'sharp ink' map at the working resolution: background-normalised ink where (a) ink > T, (b) the
    local gradient is edge-like (kills blurry bleed-through and smudges) and (c) the pixel is outside edge artefacts
    (feeder strips, spiral coil).  Returns (map, scale) with scale converting pixel counts to 100-ppi equivalents."""
    g, d = _prep(img, dpi)
    emask, _ = edge_artifact_mask(g, d)
    ink = cv2.GaussianBlur(ink_map(g, d), (0, 0), 0.6 * d / 100)
    gb = cv2.GaussianBlur(g, (0, 0), 0.7 * d / 100).astype(np.float32)
    gm = np.hypot(cv2.Sobel(gb, cv2.CV_32F, 1, 0), cv2.Sobel(gb, cv2.CV_32F, 0, 1))
    ok = (ink > T) & (gm > grad_thr * d / 100) & ~(cv2.dilate(emask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    return np.where(ok, ink, 0).astype(np.float32), (100.0 / d) ** 2


def blank_score(img: np.ndarray, dpi: float) -> float:
    """Sharp-ink mass of the page: sum of background-normalised ink over pixels that are darker than the paper by
    >12 %, have edge-like gradients, and are not scanner-edge strips / binding coil; in pixel-equivalents at 100 ppi.
    ~0 for blank sheets (dust and bleed-through give single digits to ~25), >=230 for every real labelled mark."""
    m, sc = blank_map(img, dpi)
    return float(m.sum() * sc)


def is_blank(img: np.ndarray, dpi: float, threshold: float = BLANK_THRESHOLD) -> bool:
    """True when the page carries no intentional mark (dust, bleed-through, edge shadows only)."""
    return blank_score(img, dpi) < threshold
