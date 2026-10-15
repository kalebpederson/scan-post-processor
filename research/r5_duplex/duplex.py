"""R5 prototype: duplex-aware bleed-through suppression (research code; imports scanpp read-only from a frozen snapshot).

Public API (see REPORT.md section 6):
    ink_maps(rgb, dpi)                      -> InkMaps   (50 ppi flat-field densities, ~20 ms/page)
    register_pair(mapsA, mapsB, ...)        -> PairReg | None   (pair detection + mirrored affine; None = not a pair / low confidence)
    bleed_suppression(...)                  -> float32 HxW in [0,1] (1 = remove)   [defined below]
    apply_suppression(out_rgb, s, paper)    -> uint8 HxWx3
All lengths are inches; analysis is at 50 ppi, application at native resolution.
"""
from __future__ import annotations
import os, sys
import numpy as np, cv2
_here = os.path.dirname(os.path.abspath(__file__))
_snap = os.path.join(_here, 'work', 'snap')                    # frozen copy of src/scanpp used for the study (gitignored); falls back to the repo's src/
sys.path.insert(0, _snap if os.path.isdir(_snap) else os.path.join(_here, '..', '..', 'src'))
from scanpp import clean as _clean           # estimate_background, clean_page

WS = 50.0                                      # analysis ppi

# ----------------------------------------------------------------------------- ink maps
class InkMaps:
    __slots__ = ("d", "c", "shape_native", "dpi", "r")
    def __init__(self, r, shape_native, dpi):
        self.r = r; self.d = r.max(-1); self.c = self.d - r.min(-1); self.shape_native = shape_native; self.dpi = dpi

def ink_maps(rgb: np.ndarray, dpi: float, ws: float = WS) -> InkMaps:
    """Per-channel ink density r_c = clip(1 - rgb_c/bg_c, 0, 1) at 50 ppi (bg from scanpp.clean.estimate_background with the clean_page defaults)."""
    h, w = rgb.shape[:2]
    s = ws / dpi
    small = cv2.resize(rgb, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    P = _clean.DEFAULTS
    bg = np.maximum(_clean.estimate_background(small, ws, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], ws, P["bg_open_in"]), 16.0)
    r = np.clip(1.0 - small.astype(np.float32) / bg, 0, 1)
    return InkMaps(r, (h, w), dpi)

# ----------------------------------------------------------------------------- coarse detection (translation, 50 ppi)
def _sink(d, c, real_thr=0.25, cap=0.25, edge=0.1):
    """B's faint chromatic layer: chroma where the density is below real ink (d > real_thr dilated is masked)."""
    h, w = d.shape
    real = cv2.dilate((d > real_thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    F = np.where(real, 0, np.clip(c, 0, cap)).astype(np.float32)
    F[:, :int(edge * w)] = 0; F[:, int((1 - edge) * w):] = 0
    return F

def _source(c, cap=0.6):
    return np.ascontiguousarray(np.clip(c, 0, cap).astype(np.float32)[:, ::-1])     # mirrored

def _match(F, S, R):
    h, w = F.shape
    T = np.ascontiguousarray(F[R:h - R, R:w - R])
    if T.std() < 1e-4 or S.std() < 1e-4: return None
    res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
    i = np.unravel_index(res.argmax(), res.shape)
    r2 = res.copy(); y0, x0 = i
    r2[max(0, y0 - 4):y0 + 5, max(0, x0 - 4):x0 + 5] = -1
    return dict(z=float((res.max() - res.mean()) / (res.std() + 1e-9)), peak=float(res.max()), second=float(r2.max()),
                dx=int(i[1] - R), dy=int(i[0] - R), border=bool(i[0] in (0, res.shape[0] - 1) or i[1] in (0, res.shape[1] - 1)))

def coarse(mA: InkMaps, mB: InkMaps, rng_in=1.0):
    """Both directions. 'a' = A receives bleed from mirrored B: A(x,y) ~ Bm(x+dx, y+dy); 'b' = B receives from mirrored A."""
    R = int(rng_in * WS)
    h = min(mA.d.shape[0], mB.d.shape[0]); w = min(mA.d.shape[1], mB.d.shape[1])
    if h < 4 * R or w < 4 * R: return None, None
    dA, cA, dB, cB = mA.d[:h, :w], mA.c[:h, :w], mB.d[:h, :w], mB.c[:h, :w]
    a = _match(_sink(dA, cA), _source(cB), R)
    b = _match(_sink(dB, cB), _source(cA), R)
    return a, b

# ----------------------------------------------------------------------------- fine registration (ECC affine, 50 ppi)
def _ncc(a, b, m):
    a = a[m] - a[m].mean(); b = b[m] - b[m].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))

def refine(sink, src, dx, dy, blur=1.2, iters=100):
    """ECC affine: W (2x3, 50 ppi, crop coordinates) such that warpAffine(src, W, INVERSE_MAP) ~ sink.  Returns (W, ncc, rho) (rho None if ECC failed)."""
    h, w = sink.shape
    valid = np.zeros((h, w), np.uint8); valid[int(.04 * h):int(.96 * h), int(.1 * w):int(.9 * w)] = 1
    Fb = cv2.GaussianBlur(sink, (0, 0), blur); Sb = cv2.GaussianBlur(src, (0, 0), blur)
    W0 = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    al = lambda W: cv2.warpAffine(Sb, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    n0 = _ncc(Fb, al(W0), valid > 0)
    W, rho = W0, None
    try:
        rho, W2 = cv2.findTransformECC(Fb, Sb, W0.copy(), cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5), valid, 3)
        n2 = _ncc(Fb, al(W2), valid > 0)
        if n2 >= n0 and abs(abs(W2[0, 0]) - 1) < 0.012 and abs(abs(W2[1, 1]) - 1) < 0.012 and abs(W2[0, 1]) < 0.03 and abs(W2[1, 0]) < 0.03:
            W, n0 = W2, n2
    except cv2.error:
        pass
    if W is W0:      # affine rejected (not rigid enough): rigid (rotation + translation) ECC
        try:
            rho_e, W3 = cv2.findTransformECC(Fb, Sb, W0.copy(), cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-5), valid, 3)
            n3 = _ncc(Fb, al(W3), valid > 0)
            if n3 >= n0 and abs(W3[0, 1]) < 0.03: W, n0, rho = W3, n3, rho_e
        except cv2.error:
            pass
    return W.astype(np.float64), n0, (None if rho is None else float(rho))

class PairReg:
    """Registration of page B (mirrored) onto page A.
    M_ab (2x3, native pixels): A native (x,y) -> B native (x',y') of the same paper point (includes the mirror):
        B_in_A = cv2.warpAffine(B, M_ab, (wA, hA), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    M_ba is the inverse (A resampled into B's frame)."""
    def __init__(self, **kw): self.__dict__.update(kw)

def _native_matrix(W50, wc50, dpi_A, dpi_B):
    """W50: crop-space map sink(A) -> Bm; returns native A->B incl. un-mirroring. wc50 = common width at 50 ppi."""
    sA = dpi_A / WS; sB = dpi_B / WS
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

def register_pair(mA: InkMaps, mB: InkMaps, coarse_gap_min=0.05, rng_in=1.0):
    """Detect whether A,B are the two sides of one sheet and register B onto A.  Returns dict(feat..., reg=PairReg|None).
    The decision is accept(feat); features are kept so thresholds can be swept."""
    a, b = coarse(mA, mB, rng_in)
    feat = dict(a=a, b=b, ga=_gap(a), gb=_gap(b), reg=None)
    if a is None or b is None: return feat
    feat["gmax"] = max(feat["ga"], feat["gb"])
    if feat["gmax"] < coarse_gap_min: return feat
    h = min(mA.d.shape[0], mB.d.shape[0]); w = min(mA.d.shape[1], mB.d.shape[1])
    dA, cA, dB, cB = mA.d[:h, :w], mA.c[:h, :w], mB.d[:h, :w], mB.c[:h, :w]
    res = {}
    for key, sink, src, d in (("a", _sink(dA, cA), _source(cB), a), ("b", _sink(dB, cB), _source(cA), b)):
        if _gap(d) < coarse_gap_min * 0.6: continue
        res[key] = refine(sink, src, d["dx"], d["dy"])
    if not res: return feat
    best = max(res, key=lambda k: res[k][1])
    feat["fine"] = {k: dict(ncc=v[1], rho=v[2], W=v[0].tolist()) for k, v in res.items()}
    feat["best"] = best; feat["ncc"] = res[best][1]; feat["rho"] = res[best][2]
    W = res[best][0]
    if best == "a": Mab = _native_matrix(W, w, mA.dpi, mB.dpi)
    else: Mab = _inv(_native_matrix(W, w, mB.dpi, mA.dpi))
    feat["M_ab"] = Mab
    # agreement of the two directional registrations (grid of points over A, native A pixels)
    if len(res) == 2:
        Ma = _native_matrix(res["a"][0], w, mA.dpi, mB.dpi); Mb = _inv(_native_matrix(res["b"][0], w, mB.dpi, mA.dpi))
        H, Wn = mA.shape_native; gx, gy = np.meshgrid(np.linspace(.1 * Wn, .9 * Wn, 5), np.linspace(.1 * H, .9 * H, 5))
        P = np.stack([gx.ravel(), gy.ravel(), np.ones(25)])
        feat["agree_in"] = float(np.abs((Ma @ P) - (Mb @ P)).max() / mA.dpi)
    else:
        feat["agree_in"] = None
    L = Mab[:, :2]
    feat["shift_in"] = [float(Mab[0, 2] / mA.dpi), float(Mab[1, 2] / mA.dpi)]
    feat["reg"] = PairReg(M_ab=Mab, dpi_A=mA.dpi, dpi_B=mB.dpi)
    return feat

# ----------------------------------------------------------------------------- suppression (component rule)
SUPP = dict(src_cap=0.6, blur_in=0.01, foot_thr=0.15, open_in=0.04, tol_in=0.05, ink_thr=0.04, mag_ratio=0.95, chroma_mag=0.9, grow_in=0.012,
            firm_d=0.12, neutral_rel=0.4, cos_pix=0.3, reach_in=0.06, reach_d=0.25, protect_in=0.005)

def native_density(rgb, dpi):
    """per-channel ink density planes r_c at native resolution (50 ppi flat-field background, same as clean_page's)."""
    h, w = rgb.shape[:2]
    s = WS / dpi
    small = cv2.resize(rgb, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
    P = _clean.DEFAULTS
    bg = np.maximum(_clean.estimate_background(small, WS, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], WS, P["bg_open_in"]), 16.0)
    ibg = [cv2.resize(np.ascontiguousarray(1.0 / bg[..., c]), (w, h), interpolation=cv2.INTER_LINEAR) for c in range(3)]
    return [np.clip(1.0 - cv2.multiply(p, ib, dtype=cv2.CV_32F), 0, 1) for p, ib in zip(cv2.split(rgb), ibg)]

def source_in_B_frame(rgbA, reg, shapeB, src_ws=100.0):
    """A's per-channel ink planes at src_ws, resampled into B's native frame: returns list of 3 float32 HxW (B native)."""
    mA = ink_maps(rgbA, reg.dpi_A, ws=src_ws)
    sA = src_ws / reg.dpi_A
    S_A = np.array([[sA, 0, .5 * sA - .5], [0, sA, .5 * sA - .5], [0, 0, 1]])
    M = (S_A @ np.linalg.inv(np.vstack([reg.M_ab, [0, 0, 1]])))[:2]
    h, w = shapeB
    return [cv2.warpAffine(np.ascontiguousarray(mA.r[..., c]), M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0) for c in range(3)]

def bleed_suppression(rgbA, rgbB, reg, P=SUPP, debug=False):
    """s in [0,1] (float32 HxW, B native): 1 = this pixel of B is bleed-through of A's wet ink.
    candidates G: ink pixels inside A's registered chromatic-ink footprint (dilated by the registration tolerance).
    protected (real) pixels: (a) seeds = ink outside the footprint, ink darker than the source can explain (d > mag_ratio * source d), or
    ink whose colour direction is not A's ink colour (graphite, other pens); (b) geodesic growth of the seeds along dark ink for `reach_in`
    so the part of a stroke that crosses the band stays attached to the part outside it.  s = G minus protected (+ a thin skirt)."""
    dpi = reg.dpi_B; h, w = rgbB.shape[:2]
    rA = source_in_B_frame(rgbA, reg, (h, w))
    dA = cv2.max(cv2.max(rA[0], rA[1]), rA[2]); cA = dA - cv2.min(cv2.min(rA[0], rA[1]), rA[2])
    S = cv2.GaussianBlur(np.minimum(cA, P["src_cap"]), (0, 0), max(0.7, P["blur_in"] * dpi))
    # marker-grade source = chromatic ink that survives an opening wider than a pen stroke (thin-pen ghosts are the single-page cleaner's job)
    ko = max(1, int(round(P["open_in"] * dpi / 2)))
    foot = cv2.morphologyEx((S > P["foot_thr"]).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ko + 1,) * 2)) > 0
    zero = np.zeros((h, w), np.float32)
    if foot.sum() < 100: return (zero, {}) if debug else zero
    bx, by, bw, bh = cv2.boundingRect(foot.astype(np.uint8)); pad = int(0.15 * dpi)
    y0, y1 = max(0, by - pad), min(h, by + bh + pad); x0, x1 = max(0, bx - pad), min(w, bx + bw + pad)
    r = [p[y0:y1, x0:x1] for p in native_density(rgbB, dpi)]
    d = cv2.max(cv2.max(r[0], r[1]), r[2]); mn = cv2.min(cv2.min(r[0], r[1]), r[2]); ch = d - mn
    Sr = S[y0:y1, x0:x1]
    tol = max(1, int(round(P["tol_in"] * dpi)))
    ell = lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    Fd = cv2.dilate((Sr > P["foot_thr"]).astype(np.uint8), ell(tol)) > 0
    ink = d > P["ink_thr"]
    # colour direction of the source under each pixel (smoothed A planes) vs the pixel's own
    mB = (r[0] + r[1] + r[2]) / 3; mAr = [p[y0:y1, x0:x1] for p in rA]; mA3 = (mAr[0] + mAr[1] + mAr[2]) / 3
    uB = [r[c] - mB for c in range(3)]; uA = [cv2.GaussianBlur(mAr[c] - mA3, (0, 0), 0.01 * dpi) for c in range(3)]
    dot = uB[0] * uA[0] + uB[1] * uA[1] + uB[2] * uA[2]
    nB = np.sqrt(uB[0] ** 2 + uB[1] ** 2 + uB[2] ** 2) + 1e-6; nA = np.sqrt(uA[0] ** 2 + uA[1] ** 2 + uA[2] ** 2) + 1e-6
    cosp = dot / (nA * nB)
    dAx = cv2.dilate(dA[y0:y1, x0:x1], ell(tol))
    firm = d > P["firm_d"]
    ratioA = cv2.GaussianBlur(cA[y0:y1, x0:x1], (0, 0), 0.01 * dpi) / np.maximum(cv2.GaussianBlur(dA[y0:y1, x0:x1], (0, 0), 0.01 * dpi), 0.05)   # source chroma/density
    mismatch = firm & ((ch < P["neutral_rel"] * ratioA * d) | (cosp < P["cos_pix"]))
    cAx = cv2.dilate(cA[y0:y1, x0:x1], ell(tol))
    toodark = (d > P["mag_ratio"] * dAx) | (ch > P["chroma_mag"] * cAx)     # a ghost is paler than its source, both in density and in colour strength
    seeds = ink & (~Fd | mismatch | toodark)
    # geodesic growth of the seeds along dark ink (a stroke that crosses the band)
    dark = d > P["reach_d"]
    prot = seeds.astype(np.uint8)
    reach = int(round(P["reach_in"] * dpi)); step = max(1, reach // 8)
    kern = ell(step)
    for _ in range(max(1, reach // step)):
        grown = cv2.dilate(prot, kern) & (dark | seeds).astype(np.uint8)
        if (grown == prot).all(): break
        prot = grown
    prot = cv2.dilate(prot, ell(max(1, int(round(P["protect_in"] * dpi))))) > 0
    G = ink & Fd & ~prot
    gr = max(1, int(round(P["grow_in"] * dpi)))
    m = (cv2.dilate(G.astype(np.uint8), ell(gr)) > 0) & ~prot & (d > 0.012)
    s = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8)
    s = np.where(prot, 0, np.minimum(1.0, s * 1.5)).astype(np.float32)
    out = zero; out[y0:y1, x0:x1] = s
    if debug: return out, dict(roi=(y0, y1, x0, x1), S=S, n_cand=int(G.sum()), seeds=seeds, mismatch=mismatch, toodark=toodark, prot=prot, Fd=Fd, d=d, cosp=cosp, ch=ch)
    return out

def apply_suppression(out_rgb, s, paper_rgb=(255, 255, 255)):
    """blend B's cleaned output toward paper by s (uint8 in, uint8 out)."""
    if s is None or not s.any(): return out_rgb
    o = out_rgb.astype(np.float32); p = np.array(paper_rgb, np.float32)
    return np.clip(o + s[..., None] * (p - o) + .5, 0, 255).astype(np.uint8)

# ----------------------------------------------------------------------------- decision + sheet pairing
GATE = dict(coarse_gap_min=0.05, g_two=0.05, agree_in_max=0.12, ncc_two=0.20, g_one=0.06, ncc_one=0.50, chroma_px_min=400)

def chroma_px(m: InkMaps) -> int:
    """number of 50 ppi pixels with chroma > 0.15 (wet colour ink: pen / marker / highlighter)."""
    return int((m.c > 0.15).sum())

def accept(feat, mA: InkMaps, mB: InkMaps, G=GATE) -> bool:
    """Gate.  Two-way: both directions find the same registration (their own peaks agree to <= agree_in_max inches) with peak-to-second-peak gap
    >= g_two and ncc >= ncc_two.  One-way (e.g. a blank back that only carries the ghost): gap >= g_one and ncc >= ncc_one.  Either way the
    bleed SOURCE page must carry >= chroma_px_min px of colour ink.  Everything else: no pair."""
    if feat.get("ncc") is None or feat.get("reg") is None: return False
    two = feat.get("agree_in") is not None and feat["ga"] >= G["g_two"] and feat["gb"] >= G["g_two"] and feat["agree_in"] <= G["agree_in_max"] and feat["ncc"] >= G["ncc_two"]
    one = max(feat["ga"], feat["gb"]) >= G["g_one"] and feat["ncc"] >= G["ncc_one"]
    src = mB if feat["best"] == "a" else mA          # direction 'a': A receives from B
    return bool((two or one) and chroma_px(src) >= G["chroma_px_min"])

def pair_sheets(maps: list, G=GATE):
    """maps: InkMaps per page in document order (None for unreadable).  Tests every consecutive pair (i, i+1) (both parities: a doc whose blank
    backs were deleted shifts the parity), keeps accepted pairs, resolves conflicts (a page in two accepted pairs -> keep the pair with the higher
    ncc*gap only if it beats the other by 1.5x, else drop both).  Returns list of (i, i+1, feat)."""
    cand = []
    for i in range(len(maps) - 1):
        a, b = maps[i], maps[i + 1]
        if a is None or b is None: continue
        if abs(a.d.shape[1] - b.d.shape[1]) > 0.02 * a.d.shape[1] or abs(a.d.shape[0] - b.d.shape[0]) > 0.05 * a.d.shape[0]: continue
        f = register_pair(a, b, G["coarse_gap_min"])
        if accept(f, a, b, G): cand.append((i, i + 1, f))
    keep = []
    for k, (i, j, f) in enumerate(cand):
        sc = f["ncc"] * max(f["ga"], f["gb"])
        rivals = [c for c in cand if c is not cand[k] and (c[0] == j or c[1] == i)]
        if all(sc >= 1.5 * c[2]["ncc"] * max(c[2]["ga"], c[2]["gb"]) for c in rivals): keep.append((i, j, f))
    return keep
