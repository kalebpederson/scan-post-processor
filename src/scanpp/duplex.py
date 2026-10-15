"""Duplex-aware bleed-through suppression (research/r5_duplex/REPORT.md).

A coloured marker on one side of a sheet shows through, mirrored and paler, on the other side.  The single-page cleaner cannot tell such a
ghost from real ink, but the other side of the sheet can: where the partner page carries marker-grade colour ink, the same paper point
on this page is a ghost candidate unless it is darker / more saturated than the source can explain or has another colour.

    ink_maps(rgb, dpi)                   -> InkMaps      50 ppi ink density / chroma maps (what the pairing needs)
    pair_sheets(maps)                    -> [(i, i+1, feat)]   accepted sheets among consecutive pages (feat['reg'] registers i onto i+1)
    bleed_suppression(src, dst, reg)     -> float32 HxW in [0, 1] (1 = remove)   mask for page `dst` from its partner `src`
    apply_suppression(out, s, paper)     -> uint8 HxWx3  cleaned page blended toward paper

All lengths are inches; pairing is analysed at 50 ppi, suppression is applied at native resolution.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import clean as _clean

DUPLEX_VERSION = "1"        # bump when a change here can alter the output (the constants below are tuned on a corpus; see the R5 report)

WS = 50.0                   # analysis ppi


# ----------------------------------------------------------------------------- ink maps
class InkMaps:
    """Per-channel ink density `r` (HxWx3), its max `d` and chroma `c` = max - min, at 50 ppi."""
    __slots__ = ("d", "c", "shape_native", "dpi", "r")

    def __init__(self, r, shape_native, dpi):
        self.r = r
        a, b, c = cv2.split(r)                  # (cv2 reduces three planes ~15x faster than r.max(-1) / r.min(-1); the same values)
        self.d = cv2.max(cv2.max(a, b), c)
        self.c = self.d - cv2.min(cv2.min(a, b), c)
        self.shape_native = shape_native
        self.dpi = dpi


def _density(rgb: np.ndarray, dpi: float, ws: float) -> np.ndarray:
    """Per-channel ink density r_c = clip(1 - rgb_c / bg_c, 0, 1) at `ws` ppi (bg from `clean.estimate_background` with the `clean_page` defaults)."""
    h, w = rgb.shape[:2]
    s = ws / dpi
    small = cv2.resize(rgb, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    P = _clean.DEFAULTS
    bg = np.maximum(_clean.estimate_background(small, ws, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], ws, P["bg_open_in"]), 16.0)
    return np.clip(1.0 - small.astype(np.float32) / bg, 0, 1)


def ink_maps(rgb: np.ndarray, dpi: float, ws: float = WS) -> InkMaps:
    """The ink density planes at `ws` ppi with their max and chroma."""
    return InkMaps(_density(rgb, dpi, ws), rgb.shape[:2], dpi)


# ----------------------------------------------------------------------------- coarse detection (translation, 50 ppi)
def _sink(d, c, real_thr=0.25, cap=0.25, edge=0.1):
    """The receiving page's faint chromatic layer: chroma where the density is below real ink (d > real_thr, dilated, is masked)."""
    h, w = d.shape
    real = cv2.dilate((d > real_thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    F = np.where(real, 0, np.clip(c, 0, cap)).astype(np.float32)
    F[:, :int(edge * w)] = 0
    F[:, int((1 - edge) * w):] = 0
    return F


def _source(c, cap=0.6):
    return np.ascontiguousarray(np.clip(c, 0, cap).astype(np.float32)[:, ::-1])     # mirrored


def _match(F, S, R):
    h, w = F.shape
    T = np.ascontiguousarray(F[R:h - R, R:w - R])
    if T.std() < 1e-4 or S.std() < 1e-4:
        return None
    res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
    i = np.unravel_index(res.argmax(), res.shape)
    r2 = res.copy()
    y0, x0 = i
    r2[max(0, y0 - 4):y0 + 5, max(0, x0 - 4):x0 + 5] = -1
    return dict(peak=float(res.max()), second=float(r2.max()), dx=int(i[1] - R), dy=int(i[0] - R),
                border=bool(i[0] in (0, res.shape[0] - 1) or i[1] in (0, res.shape[1] - 1)))


def _coarse(mA: InkMaps, mB: InkMaps, rng_in=1.0):
    """Both directions.  'a' = A receives bleed from mirrored B: A(x,y) ~ Bm(x+dx, y+dy); 'b' = B receives from mirrored A."""
    R = int(rng_in * WS)
    h = min(mA.d.shape[0], mB.d.shape[0])
    w = min(mA.d.shape[1], mB.d.shape[1])
    if h < 4 * R or w < 4 * R:
        return None, None
    dA, cA, dB, cB = mA.d[:h, :w], mA.c[:h, :w], mB.d[:h, :w], mB.c[:h, :w]
    a = _match(_sink(dA, cA), _source(cB), R)
    b = _match(_sink(dB, cB), _source(cA), R)
    return a, b


# ----------------------------------------------------------------------------- fine registration (ECC affine, 50 ppi)
def _ncc(a, b, m):
    a = a[m] - a[m].mean()
    b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def _refine(sink, src, dx, dy, blur=1.2, iters=100):
    """ECC affine: W (2x3, 50 ppi, crop coordinates) such that warpAffine(src, W, INVERSE_MAP) ~ sink.  Falls back to rigid ECC, then to the
    plain translation.  Returns (W, ncc)."""
    h, w = sink.shape
    valid = np.zeros((h, w), np.uint8)
    valid[int(.04 * h):int(.96 * h), int(.1 * w):int(.9 * w)] = 1
    Fb = cv2.GaussianBlur(sink, (0, 0), blur)
    Sb = cv2.GaussianBlur(src, (0, 0), blur)
    W0 = np.array([[1, 0, dx], [0, 1, dy]], np.float32)

    def align(W):
        return cv2.warpAffine(Sb, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def crit():
        return (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5)

    n0 = _ncc(Fb, align(W0), valid > 0)
    W = W0
    try:
        _, W2 = cv2.findTransformECC(Fb, Sb, W0.copy(), cv2.MOTION_AFFINE, crit(), valid, 3)
        n2 = _ncc(Fb, align(W2), valid > 0)
        if n2 >= n0 and abs(abs(W2[0, 0]) - 1) < 0.012 and abs(abs(W2[1, 1]) - 1) < 0.012 and abs(W2[0, 1]) < 0.03 and abs(W2[1, 0]) < 0.03:
            W, n0 = W2, n2
    except cv2.error:
        pass
    if W is W0:      # affine rejected (not rigid enough): rigid (rotation + translation) ECC
        try:
            _, W3 = cv2.findTransformECC(Fb, Sb, W0.copy(), cv2.MOTION_EUCLIDEAN, crit(), valid, 3)
            n3 = _ncc(Fb, align(W3), valid > 0)
            if n3 >= n0 and abs(W3[0, 1]) < 0.03:
                W, n0 = W3, n3
        except cv2.error:
            pass
    return W.astype(np.float64), n0


class PairReg:
    """Registration between two pages: `M_ab` (2x3, native pixels) maps the native (x, y) of page A to the native (x', y') of page B of the same
    paper point (it includes the mirror).  `dpi_A` / `dpi_B` are the pages' resolutions."""

    def __init__(self, M_ab, dpi_A, dpi_B):
        self.M_ab = M_ab
        self.dpi_A = dpi_A
        self.dpi_B = dpi_B

    def inverse(self) -> "PairReg":
        """The same registration seen from the other page (B -> A)."""
        return PairReg(_inv(self.M_ab), self.dpi_B, self.dpi_A)


def _native_matrix(W50, wc50, dpi_A, dpi_B):
    """W50: crop-space map sink(A) -> Bm; returns native A->B incl. un-mirroring.  wc50 = common width at 50 ppi."""
    sA = dpi_A / WS
    sB = dpi_B / WS
    # A native p -> A 50ppi p50 = (p+.5)/sA-.5 ; Bm50 = W50 p50 ; B50 = (wc50-1 - Bm50x, Bm50y) ; B native = (B50+.5)*sB-.5
    S_A = np.array([[1 / sA, 0, .5 / sA - .5], [0, 1 / sA, .5 / sA - .5], [0, 0, 1]])
    W = np.vstack([W50, [0, 0, 1]])
    Mir = np.array([[-1, 0, wc50 - 1], [0, 1, 0], [0, 0, 1]])
    S_B = np.array([[sB, 0, .5 * sB - .5], [0, sB, .5 * sB - .5], [0, 0, 1]])
    return (S_B @ Mir @ W @ S_A)[:2]


def _inv(M):
    return np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]


def _gap(d):
    return 0.0 if d is None or d["border"] else d["peak"] - d["second"]


def register_pair(mA: InkMaps, mB: InkMaps, coarse_gap_min=0.05, rng_in=1.0) -> dict:
    """Detect whether A, B are the two sides of one sheet and register them.  Returns the features the decision (`accept`) is based on:
    ga, gb (peak uniqueness per direction), ncc, best ('a' | 'b'), agree_in (inches, None for one-way) and, when a registration was found,
    reg = PairReg (A -> B)."""
    a, b = _coarse(mA, mB, rng_in)
    feat = dict(ga=_gap(a), gb=_gap(b), reg=None)
    if a is None or b is None:
        return feat
    if max(feat["ga"], feat["gb"]) < coarse_gap_min:
        return feat
    h = min(mA.d.shape[0], mB.d.shape[0])
    w = min(mA.d.shape[1], mB.d.shape[1])
    dA, cA, dB, cB = mA.d[:h, :w], mA.c[:h, :w], mB.d[:h, :w], mB.c[:h, :w]
    res = {}
    for key, sink, src, d in (("a", _sink(dA, cA), _source(cB), a), ("b", _sink(dB, cB), _source(cA), b)):
        if _gap(d) < coarse_gap_min * 0.6:
            continue
        res[key] = _refine(sink, src, d["dx"], d["dy"])
    if not res:
        return feat
    best = max(res, key=lambda k: res[k][1])
    feat["best"] = best
    feat["ncc"] = res[best][1]
    W = res[best][0]
    if best == "a":
        Mab = _native_matrix(W, w, mA.dpi, mB.dpi)
    else:
        Mab = _inv(_native_matrix(W, w, mB.dpi, mA.dpi))
    # agreement of the two directional registrations (grid of points over A, native A pixels)
    feat["agree_in"] = None
    if len(res) == 2:
        Ma = _native_matrix(res["a"][0], w, mA.dpi, mB.dpi)
        Mb = _inv(_native_matrix(res["b"][0], w, mB.dpi, mA.dpi))
        H, Wn = mA.shape_native
        gx, gy = np.meshgrid(np.linspace(.1 * Wn, .9 * Wn, 5), np.linspace(.1 * H, .9 * H, 5))
        P = np.stack([gx.ravel(), gy.ravel(), np.ones(25)])
        feat["agree_in"] = float(np.abs((Ma @ P) - (Mb @ P)).max() / mA.dpi)
    feat["reg"] = PairReg(Mab, mA.dpi, mB.dpi)
    return feat


# ----------------------------------------------------------------------------- suppression
SUPP = dict(src_cap=0.6, blur_in=0.01, foot_thr=0.15, open_in=0.04, tol_in=0.05, ink_thr=0.04, mag_ratio=0.95, chroma_mag=0.9, grow_in=0.012,
            firm_d=0.12, neutral_rel=0.4, cos_pix=0.3, reach_in=0.06, reach_d=0.25, protect_in=0.005)


def _native_density(rgb, dpi):
    """Per-channel ink density planes r_c at native resolution (50 ppi flat-field background, same as clean_page's)."""
    h, w = rgb.shape[:2]
    s = WS / dpi
    small = cv2.resize(rgb, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    P = _clean.DEFAULTS
    bg = np.maximum(_clean.estimate_background(small, WS, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], WS, P["bg_open_in"]), 16.0)
    ibg = [cv2.resize(np.ascontiguousarray(1.0 / bg[..., c]), (w, h), interpolation=cv2.INTER_LINEAR) for c in range(3)]
    return [np.clip(1.0 - cv2.multiply(p, ib, dtype=cv2.CV_32F), 0, 1) for p, ib in zip(cv2.split(rgb), ibg)]


def _source_in_dst_frame(src_rgb, reg, shape, src_ws=100.0):
    """The source page's per-channel ink planes at `src_ws`, resampled into the destination's native frame: 3 float32 HxW arrays."""
    rS = _density(src_rgb, reg.dpi_A, src_ws)
    sA = src_ws / reg.dpi_A
    S_A = np.array([[sA, 0, .5 * sA - .5], [0, sA, .5 * sA - .5], [0, 0, 1]])
    M = (S_A @ np.linalg.inv(np.vstack([reg.M_ab, [0, 0, 1]])))[:2]
    h, w = shape
    return [cv2.warpAffine(np.ascontiguousarray(rS[..., c]), M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
            for c in range(3)]


def _touches_ink(cleaned: np.ndarray, Fd: np.ndarray, grow: int, paper_rgb) -> bool:
    """False only if `cleaned` is paper at every pixel the suppression mask could be non-zero at: the mask is the `grow` dilation of ink inside
    the dilated footprint `Fd`, then blurred (sigma 0.8: a 5 px kernel).  The reach is taken one pixel wider than that for the discretised
    structuring elements."""
    reach = cv2.dilate(Fd.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (grow + 3) + 1,) * 2)) > 0
    return bool(reach.any() and (cleaned[reach] != np.array(paper_rgb, np.uint8)).any())


def bleed_suppression(src_rgb: np.ndarray, dst_rgb: np.ndarray, reg: PairReg, P: dict = SUPP, cleaned: np.ndarray | None = None,
                      paper_rgb=(255, 255, 255)) -> np.ndarray:
    """s in [0, 1] (float32 HxW, `dst_rgb` native): 1 = this pixel of the destination page is bleed-through of the source page's wet ink.
    `reg` registers the source onto the destination (M_ab: source native -> destination native, dpi_A = source, dpi_B = destination).

    Candidates G: ink pixels inside the source's registered marker-grade chromatic footprint (dilated by the registration tolerance).
    Protected (real) pixels: (a) seeds = ink outside the footprint, ink darker than the source can explain (d > mag_ratio * source d), ink more
    saturated than the source, or ink whose colour direction is not the source ink's (graphite, other pens); (b) geodesic growth of the seeds
    along dark ink for `reach_in`, so the part of a stroke that crosses the band stays attached to the part outside it.
    s = G minus protected (+ a thin skirt).

    `cleaned` (optional, the destination after the single-page cleaner, `paper_rgb` its paper colour) lets the function stop early, with the
    same result as far as `apply_suppression` is concerned: the mask can only be non-zero within `grow` + a few pixels of the source's (dilated) footprint, and where
    `cleaned` is paper everywhere in there the blend changes nothing, so zeros are returned without analysing the destination."""
    dpi = reg.dpi_B
    h, w = dst_rgb.shape[:2]
    rA = _source_in_dst_frame(src_rgb, reg, (h, w))
    dA = cv2.max(cv2.max(rA[0], rA[1]), rA[2])
    cA = dA - cv2.min(cv2.min(rA[0], rA[1]), rA[2])
    S = cv2.GaussianBlur(np.minimum(cA, P["src_cap"]), (0, 0), max(0.7, P["blur_in"] * dpi))
    # marker-grade source = chromatic ink that survives an opening wider than a pen stroke (thin-pen ghosts are the single-page cleaner's job)
    ko = max(1, int(round(P["open_in"] * dpi / 2)))
    foot = cv2.morphologyEx((S > P["foot_thr"]).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ko + 1,) * 2)) > 0
    zero = np.zeros((h, w), np.float32)
    if foot.sum() < 100:
        return zero
    bx, by, bw, bh = cv2.boundingRect(foot.astype(np.uint8))
    pad = int(0.15 * dpi)
    y0, y1 = max(0, by - pad), min(h, by + bh + pad)
    x0, x1 = max(0, bx - pad), min(w, bx + bw + pad)
    Sr = S[y0:y1, x0:x1]
    tol = max(1, int(round(P["tol_in"] * dpi)))

    def ell(k):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))

    Fd = cv2.dilate((Sr > P["foot_thr"]).astype(np.uint8), ell(tol)) > 0
    if cleaned is not None and not _touches_ink(cleaned[y0:y1, x0:x1], Fd, max(1, int(round(P["grow_in"] * dpi))), paper_rgb):
        return zero
    r = [p[y0:y1, x0:x1] for p in _native_density(dst_rgb, dpi)]
    d = cv2.max(cv2.max(r[0], r[1]), r[2])
    mn = cv2.min(cv2.min(r[0], r[1]), r[2])
    ch = d - mn
    ink = d > P["ink_thr"]
    # colour direction of the source under each pixel (smoothed source planes) vs the pixel's own
    mB = (r[0] + r[1] + r[2]) / 3
    mAr = [p[y0:y1, x0:x1] for p in rA]
    mA3 = (mAr[0] + mAr[1] + mAr[2]) / 3
    uB = [r[c] - mB for c in range(3)]
    uA = [cv2.GaussianBlur(mAr[c] - mA3, (0, 0), 0.01 * dpi) for c in range(3)]
    dot = uB[0] * uA[0] + uB[1] * uA[1] + uB[2] * uA[2]
    nB = np.sqrt(uB[0] ** 2 + uB[1] ** 2 + uB[2] ** 2) + 1e-6
    nA = np.sqrt(uA[0] ** 2 + uA[1] ** 2 + uA[2] ** 2) + 1e-6
    cosp = dot / (nA * nB)
    dAx = cv2.dilate(dA[y0:y1, x0:x1], ell(tol))
    firm = d > P["firm_d"]
    ratioA = cv2.GaussianBlur(cA[y0:y1, x0:x1], (0, 0), 0.01 * dpi) / np.maximum(cv2.GaussianBlur(dA[y0:y1, x0:x1], (0, 0), 0.01 * dpi), 0.05)   # source chroma / density
    mismatch = firm & ((ch < P["neutral_rel"] * ratioA * d) | (cosp < P["cos_pix"]))
    cAx = cv2.dilate(cA[y0:y1, x0:x1], ell(tol))
    toodark = (d > P["mag_ratio"] * dAx) | (ch > P["chroma_mag"] * cAx)     # a ghost is paler than its source, both in density and in colour strength
    seeds = ink & (~Fd | mismatch | toodark)
    # geodesic growth of the seeds along dark ink (a stroke that crosses the band)
    dark = d > P["reach_d"]
    prot = seeds.astype(np.uint8)
    reach = int(round(P["reach_in"] * dpi))
    step = max(1, reach // 8)
    kern = ell(step)
    for _ in range(max(1, reach // step)):
        grown = cv2.dilate(prot, kern) & (dark | seeds).astype(np.uint8)
        if (grown == prot).all():
            break
        prot = grown
    prot = cv2.dilate(prot, ell(max(1, int(round(P["protect_in"] * dpi))))) > 0
    G = ink & Fd & ~prot
    gr = max(1, int(round(P["grow_in"] * dpi)))
    m = (cv2.dilate(G.astype(np.uint8), ell(gr)) > 0) & ~prot & (d > 0.012)
    s = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
    s = np.where(prot, 0, np.minimum(1.0, s * 1.5)).astype(np.float32)
    zero[y0:y1, x0:x1] = s
    return zero


def apply_suppression(out_rgb: np.ndarray, s: np.ndarray | None, paper_rgb=(255, 255, 255)) -> np.ndarray:
    """Blend the cleaned page toward paper by s (uint8 in, uint8 out; the input itself if there is nothing to remove)."""
    if s is None or not s.any():
        return out_rgb
    o = out_rgb.astype(np.float32)
    p = np.array(paper_rgb, np.float32)
    return np.clip(o + s[..., None] * (p - o) + .5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- decision + sheet pairing
GATE = dict(coarse_gap_min=0.05, g_two=0.05, agree_in_max=0.12, ncc_two=0.20, g_one=0.06, ncc_one=0.50, chroma_px_min=400)


def _chroma_px(m: InkMaps) -> int:
    """Number of 50 ppi pixels with chroma > 0.15 (wet colour ink: pen / marker / highlighter)."""
    return int((m.c > 0.15).sum())


def accept(feat: dict, mA: InkMaps, mB: InkMaps, G: dict = GATE) -> bool:
    """Gate.  Two-way: both directions find the same registration (their own peaks agree to <= agree_in_max inches) with peak-to-second-peak gap
    >= g_two and ncc >= ncc_two.  One-way (e.g. a blank back that only carries the ghost): gap >= g_one and ncc >= ncc_one.  Either way the
    bleed SOURCE page must carry >= chroma_px_min px of colour ink.  Everything else: no pair."""
    if feat.get("ncc") is None or feat.get("reg") is None:
        return False
    two = (feat.get("agree_in") is not None and feat["ga"] >= G["g_two"] and feat["gb"] >= G["g_two"]
           and feat["agree_in"] <= G["agree_in_max"] and feat["ncc"] >= G["ncc_two"])
    one = max(feat["ga"], feat["gb"]) >= G["g_one"] and feat["ncc"] >= G["ncc_one"]
    src = mB if feat["best"] == "a" else mA          # direction 'a': A receives from B
    return bool((two or one) and _chroma_px(src) >= G["chroma_px_min"])


def test_pair(a, b, G: dict = GATE):
    """The accepted-pair features of consecutive pages (a, b) = (maps[i], maps[i+1]), or None if they are not two sides of one sheet."""
    if a is None or b is None:
        return None
    if abs(a.d.shape[1] - b.d.shape[1]) > 0.02 * a.d.shape[1] or abs(a.d.shape[0] - b.d.shape[0]) > 0.05 * a.d.shape[0]:
        return None
    if _chroma_px(a) < G["chroma_px_min"] and _chroma_px(b) < G["chroma_px_min"]:
        return None                  # (`accept` needs one of the two to carry colour ink: skip the registration, the expensive part)
    f = register_pair(a, b, G["coarse_gap_min"])
    return f if accept(f, a, b, G) else None


def _score(f: dict) -> float:
    return f["ncc"] * max(f["ga"], f["gb"])


def keeps_pair(cand: dict, i: int) -> bool:
    """Conflict resolution for the accepted pair (i, i+1): `cand` maps i -> features for the accepted pairs.  A page in two accepted pairs keeps
    the pair with the higher ncc*gap only if it beats the other by 1.5x, else both are dropped.  Needs the candidates i-1 .. i+1 only."""
    return all(r is None or _score(cand[i]) >= 1.5 * _score(r) for r in (cand.get(i - 1), cand.get(i + 1)))


def pair_sheets(maps: list, G: dict = GATE) -> list:
    """maps: InkMaps per page in document order (None for unreadable).  Tests every consecutive pair (i, i+1) (both parities: a document whose
    blank backs were deleted shifts the parity), keeps accepted pairs, resolves conflicts (`keeps_pair`).  Returns a list of (i, i+1, feat)."""
    cand = {}
    for i in range(len(maps) - 1):
        if (f := test_pair(maps[i], maps[i + 1], G)) is not None:
            cand[i] = f
    return [(i, i + 1, f) for i, f in cand.items() if keeps_pair(cand, i)]
