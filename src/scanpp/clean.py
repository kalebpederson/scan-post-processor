"""Scan cleaning: flat-field background division + hysteresis/sharpness soft ink gate + hue-preserving faint-ink boost.

Ported from research/r2_cleaning/clean.py (default gate "hyst"; dead experimental gates/branches removed), plus halftone handling
(light halftoned / thin print was fragmented by the per-pixel gate: see DEFAULTS pk_* / ht_* and bench/halftone/).

    clean_page(rgb, dpi, paper="flatten") -> uint8 HxWx3

Pipeline (all lengths in inches, so results are dpi-independent; native resolution in, native resolution out)
 1. Background (paper colour + illumination) per channel on a ~50 dpi copy: small grey opening -> local max (grey
    dilation, window `bg_win_in`) -> guard (never below `blob_guard` x the median background of the surrounding
    `guard_win_in` window) -> Gaussian blur.  Works for coloured paper and uneven lighting.
 2. r_c = max(0, 1 - rgb_c/bg_c) ("ink density" per channel; paper = 0).  d = max_c r_c, chroma = max r - min r.
 3. soft gate alpha in [0,1] = smoothstep((u - t_lo)/(t_hi - t_lo)) on u = blur(d) + sharp_w*|grad blur(d)|, with a
    page-adaptive noise floor and hysteresis (isolated weak components need the higher threshold t_hi_unsup).
 4. ink tone: dk_c = clip(r_c/(1-bp) * k, 0, 1), k boosts faint grey ink with a common factor on all channels.
 5. out = 1 - alpha*dk (paper -> pure white).  paper="keep": out *= global paper colour (uniform tint).
Content-quality additions (see DEFAULTS line_* / neutral_*, tests/test_quality.py):
 * line support: faint word-sized components in the line zone of supported words are supported too (a light-gray printed line that
   fades along its length used to keep only its darkest words);
 * neutral-ink snap: graphite / black ink that kept a tint of the paper (specular reflection, scanner black offset) becomes exactly
   neutral, and the chroma-only fringe band along long rules is dropped; real pen colours are untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

DEFAULTS = dict(bg_win_in=0.35, guard_win_in=1.5, blob_guard=0.8, bg_open_in=0.06, bg_work_dpi=50.0,
                t_lo=0.02, t_hi=0.14, bp=0.16, gamma=1.0,
                t_seed=0.25, t_hi_unsup=0.12, comp_min_in2=0.5, denoise_in=0.006,
                sharp_w=0.5, sharp_sigma_in=0.0033, noise_k=3.0,
                faint_gain=3.0, faint_lo=0.05, faint_hi=0.25, faint_chroma_lo=0.08, faint_chroma_hi=0.22,
                # halftone / thin light print.  desc_sigma_in = sigma of the lightly descreened density (about one halftone pitch).
                # pk_seed / pk_gain (0 disables): a descreened core >= pk_seed supports its connected component (hysteresis), and
                # inside the gate mask the gate density is at least pk_gain * descreened density (thin or dotted light strokes keep
                # their darkness while bleed, which is never that dark, is unaffected).  ht_tone: in regions with only light ink
                # (no dark ink within ht_loc_in; weight falls 1 -> 0 as the local max of the denoised density goes ht_w_lo ->
                # ht_w_hi) the tone uses the descreened density so halftone dots form continuous strokes.
                desc_sigma_in=0.0025, pk_scope="light", pk_seed=0.25, pk_gain=0.9, ht_tone=True, ht_w_lo=0.16, ht_w_hi=0.36, ht_loc_in=0.03,
                # line support: a faint word next to supported words of the same text line (printed light-gray text fades along a line,
                # so only its darkest words reach t_seed) is supported too.  line_in / line_v_in = horizontal / vertical reach (inches)
                # of the zone around supported (or peak denoised density >= line_src) word-sized components (bbox <= word_h_in x word_w_in; long rules and big shapes do not
                # count); an unsupported word-sized component whose centroid is in the zone and whose peak denoised density is >=
                # line_min is supported.  line_in = 0 disables.
                line_in=0.5, line_v_in=0.05, line_min=0.12, line_src=0.2, line_open=0.5, line_min2=0.07, line_raw2=0.16, line_thin=2.0, line_in2=0.12, word_h_in=0.4, word_w_in=1.5,
                # neutral-ink snap (colour casts on coloured paper).  Works on the paper-tint-compensated ink density r_c (flat-field
                # division).  Graphite and black print are specular / have a scanner black offset, so after the division they keep a faint
                # tint of the paper; JPEG-subsampled chroma also leaves coloured fringes beside black strokes.  A cast can only come from
                # the paper: the allowed chroma scales with the paper's own chroma pc = (max bg - min bg) / max bg.  A pixel becomes exactly
                # neutral (r_c := max r, the darkness is kept) with weight 1 - smoothstep((chroma - neutral_c0 - neutral_c_k * pc) / neutral_c_w)
                # (chroma = max r - min r; faint grey ink), or, when dense (density >= neutral_core_d), over chroma / density with
                # neutral_r0 + neutral_r_k * pc (black ink, density >= neutral_black_d, with a black-level offset).  On white paper (pc = 0) only ratio < ~0.1 is neutral:
                # dark blue / burgundy / green pens (ratio 0.2-0.26 on white paper) keep their colour.  Pixels fainter than neutral_edge_d
                # within neutral_reach_in of a neutral core (and not of a coloured core: ratio > neutral_color) in long thin chroma-only
                # bands are fringe: their ink density becomes min r.  neutral_snap=False disables.
                neutral_snap=True, neutral_c0=0.02, neutral_c_k=0.15, neutral_c_w=0.04, neutral_r0=0.08, neutral_r_k=0.9, neutral_r_w=0.08, neutral_core_d=0.5, neutral_black_d=0.8, neutral_black_reach_in=0.015,
                neutral_color=0.45, neutral_reach_in=0.045, neutral_area_in=0.07, neutral_edge_d=0.2, neutral_edge_ratio=0.8, neutral_edge_mean=0.06, neutral_run_in=0.2, neutral_d_lo=0.10, neutral_d_hi=0.20)


def estimate_background(rgb: np.ndarray, dpi: float, bg_win_in: float, guard_win_in: float, blob_guard: float,
                        bg_work_dpi: float, bg_open_in: float = 0.0) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max(1.0, dpi / bg_work_dpi)
    sw, sh = max(8, int(round(w / s))), max(8, int(round(h / s)))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(bg_win_in * bg_work_dpi)) | 1)
    if bg_open_in > 0:
        ko = max(2, int(round(bg_open_in * bg_work_dpi)))
        small = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if blob_guard > 0:
        kg = max(5, int(round(guard_win_in * bg_work_dpi)) | 1)
        med = cv2.medianBlur(cv2.resize(bg, (max(8, sw // 2), max(8, sh // 2)), interpolation=cv2.INTER_AREA),
                             max(3, (kg // 2) | 1))
        med = cv2.resize(med, (sw, sh), interpolation=cv2.INTER_LINEAR)
        bg = np.maximum(bg, (med.astype(np.float32) * blob_guard).astype(np.uint8))
    return cv2.GaussianBlur(bg.astype(np.float32), (0, 0), k / 2.5)


def _nonzero(mask_u8: np.ndarray) -> np.ndarray:
    """Flat indices of the non-zero entries of a uint8 mask (viewing a 0/1 copy as bool is ~2.5x faster)."""
    return np.flatnonzero(cv2.min(mask_u8, 1).view(np.bool_))


def _smoothstep(x: np.ndarray) -> np.ndarray:   # in place on a float32 array
    np.clip(x, 0, 1, out=x)
    t = 3.0 - 2.0 * x
    x *= x
    x *= t
    return x


def _light_lowres(bq: np.ndarray, h: int, w: int, dpi: float, P: dict, sc: int) -> np.ndarray:
    """Low-res (1/sc) weight of the descreened tone: 1 where there is only light ink within ht_loc_in, falling to 0 as the
    local max of the denoised density goes ht_w_lo -> ht_w_hi (dark ink / strong strokes keep their raw tone)."""
    sm = np.ascontiguousarray(bq[sc // 2::sc, sc // 2::sc][:h // sc, :w // sc])      # the denoised map is smooth: subsample
    kk = max(1, int(round(P["ht_loc_in"] * dpi / sc))) * 2 + 1
    dmax = 1.0 - cv2.erode(sm, np.ones((kk, kk), np.uint8))
    return 1.0 - _smoothstep((dmax - P["ht_w_lo"]) * (1.0 / (P["ht_w_hi"] - P["ht_w_lo"])))


def _sample_lowres(wl: np.ndarray, idx: np.ndarray, w: int, sc: int) -> np.ndarray:
    i = idx.astype(np.int32) if idx.size and idx[-1] < 2**31 - 1 else idx
    y = i // w
    xx = i - y * w
    y //= sc
    xx //= sc
    np.minimum(y, wl.shape[0] - 1, out=y)
    np.minimum(xx, wl.shape[1] - 1, out=xx)
    return wl.ravel()[y * wl.shape[1] + xx]


def _up_nearest(a: np.ndarray, h: int, w: int, sc: int) -> np.ndarray:
    """Blocky sc-times upsample of a low-res map, cropped / edge-padded to (h, w) (= a[y // sc, x // sc])."""
    big = np.repeat(np.repeat(a, sc, 0), sc, 1)
    return np.pad(big, ((0, max(0, h - big.shape[0])), (0, max(0, w - big.shape[1]))), mode="edge")[:h, :w]


def _comp_stats(labf, mi, w, nl):
    """Bounding-box width / height and centroid (x, y) of every component from the labels at the mask pixels `mi` (flat, ascending).
    Equal to cv2.connectedComponentsWithStats' CC_STAT_WIDTH / HEIGHT and centroids (label 0 excluded) at a fraction of its cost (the
    stats pass of OpenCV is ~6x slower than the labelling itself): the mask is reduced to its horizontal runs first."""
    wd, ht = np.zeros(nl, np.int64), np.zeros(nl, np.int64)
    cent = np.zeros((nl, 2), np.float64)
    n = mi.size
    if n == 0:
        return wd, ht, cent
    brk = np.empty(n, bool)
    brk[0] = True
    np.not_equal(np.diff(mi), 1, out=brk[1:])
    brk[1:] |= (mi[1:] % w) == 0                  # a run does not continue into the next row
    st = np.flatnonzero(brk)
    s = mi[st]
    e = mi[np.concatenate((st[1:], [n])) - 1]
    y = s // w
    x0 = s - y * w
    x1 = e - y * w
    L = labf[s]                                   # the pixels of a run (8-connected) share one label
    o = np.argsort(L, kind="stable")              # stable: runs of one label stay in raster order (y non-decreasing)
    Ls, x0o, x1o, yo = L[o], x0[o], x1[o], y[o]
    g = np.flatnonzero(np.concatenate(([True], Ls[1:] != Ls[:-1])))
    ge = np.concatenate((g[1:], [Ls.size])) - 1
    ids = Ls[g]
    wd[ids] = np.maximum.reduceat(x1o, g) - np.minimum.reduceat(x0o, g) + 1
    ht[ids] = yo[ge] - yo[g] + 1
    ln = x1 - x0 + 1
    area = np.maximum(np.bincount(L, weights=ln, minlength=nl), 1.0)
    cent[:, 0] = np.bincount(L, weights=(x0 + x1) * ln // 2, minlength=nl) / area
    cent[:, 1] = np.bincount(L, weights=y * ln, minlength=nl) / area
    return wd, ht, cent


def _line_support(has, stats, cent, labf, mi, dm_mi, raw_mi, h, w, dpi, P, sc=4, Lmi=None):
    """Extend `has` (per-label support) to faint word-sized components that sit in the line zone of supported words (see DEFAULTS).
    `stats` is the (nl, 5) cv2 stats array (with `cent` the centroids) or None: then they are derived from the labels.
    `Lmi` = labf[mi] if the caller has it."""
    nl = has.size
    if Lmi is None:
        Lmi = labf[mi]
    if P["line_in"] <= 0 or nl < 2:
        return has, None
    wmax, hmax = P["word_w_in"] * dpi, P["word_h_in"] * dpi
    if stats is None:
        wd, ht, cent = _comp_stats(labf, mi, w, nl)
    else:
        wd, ht = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    word = (wd <= wmax) & (ht <= hmax)
    word[0] = False
    strong = np.zeros(nl, bool)
    strong[Lmi[dm_mi >= P["line_src"]]] = True
    src = (has | strong) & word
    cand = ~has & word
    if not src.any() or not cand.any():
        return has, None
    pk = np.zeros(nl, np.float32)               # peak denoised density per component, only needed for the candidates
    pr = np.zeros(nl, np.float32)               # peak raw density
    sel = cand[Lmi]
    np.maximum.at(pk, Lmi[sel], dm_mi[sel])
    np.maximum.at(pr, Lmi[sel], raw_mi[sel])
    # tier 1: peak denoised density >= line_min, reach line_in; tier 2 (very faint but thin, i.e. print not bleed-through: the raw peak
    # is far above the denoised one): reach line_in2 only
    t1 = pk >= P["line_min"]
    t2 = (pk >= P["line_min2"]) & (pr >= P["line_raw2"]) & (pr >= P["line_thin"] * pk)
    cand &= t1 | t2
    if not cand.any():
        return has, None
    ky = 2 * max(1, int(round(P["line_v_in"] * dpi / sc))) + 1
    ctr = (np.nan_to_num(cent) / sc).astype(np.int64)
    ctr[:, 0] = np.minimum(ctr[:, 0], max(1, w // sc) - 1)
    ctr[:, 1] = np.minimum(ctr[:, 1], max(1, h // sc) - 1)
    pm = np.zeros(h * w, np.uint8)
    added = np.zeros(nl, bool)
    pm[mi[src[Lmi]]] = 1
    for rnd in range(10):         # zones chain along the line: words added in one round are sources in the next
        if not (cand & ~added).any():
            break
        small = (cv2.resize(pm.reshape(h, w), (max(1, w // sc), max(1, h // sc)), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        new = np.zeros(nl, bool)
        for tsel, reach in ((t1, P["line_in"]), (t2, P["line_in2"])):
            kx = 2 * max(1, int(round(reach * dpi / sc))) + 1
            zone = cv2.dilate(small, np.ones((ky, kx), np.uint8))
            new |= cand & ~added & tsel & (zone[ctr[:, 1], ctr[:, 0]] > 0)
        if not new.any():
            break
        added |= new
        pm[mi[new[Lmi]]] = 1
    return has | added, added


def _paper_colour_bg(bgs):
    """Global paper colour: median of the brighter half of the background map (the local map follows large inked areas, so it
    cannot be used for the paper's own chroma)."""
    lum = bgs[..., 0] + bgs[..., 1] + bgs[..., 2]         # == bgs.sum(-1)
    sel = lum >= np.percentile(lum, 50)
    return np.array([np.median(np.ascontiguousarray(bgs[..., c])[sel]) for c in range(3)], dtype=bgs.dtype)


def _neutral_weight(q, dpi: float, P: dict, bgs, pcol=None):
    """Weight in [0, 1] (float32 plane) of the neutral-ink snap, from the transmittance planes q_c (see DEFAULTS neutral_*)."""
    if min(q[0].shape[:2]) < 2:          # degenerate sizes: OpenCV treats 1xN arrays as scalars / column vectors
        return np.zeros(q[0].shape, np.float32), None
    qc = [cv2.min(c, 1.0) for c in q]
    qmn = cv2.min(cv2.min(qc[0], qc[1]), qc[2])
    qmx = cv2.max(cv2.max(qc[0], qc[1]), qc[2])
    dn = cv2.subtract(1.0, qmn)
    chrom = cv2.subtract(qmx, qmn)
    ratio = cv2.divide(chrom, np.maximum(dn, 0.02))
    # chroma of the page's paper (brighter half of the background map; the local map follows large inked areas, so it cannot be used)
    if pcol is None:
        pcol = _paper_colour_bg(bgs)
    pc = float((pcol.max() - pcol.min()) / max(pcol.max(), 1.0))
    clim = np.float32(pc * P["neutral_c_k"] + P["neutral_c0"])
    wn = cv2.subtract(1.0, _smoothstep((chrom - clim) * (1.0 / P["neutral_c_w"])))
    core = dn >= P["neutral_core_d"]
    rlim = np.float32(pc * P["neutral_r_k"] + P["neutral_r0"])
    wr = cv2.subtract(1.0, _smoothstep((ratio - rlim) * (1.0 / P["neutral_r_w"])))
    # only (nearly) black ink and its anti-aliased edges: a dark navy pen on yellow paper has the same ratio as black ink on blue paper
    blk = (dn >= P["neutral_black_d"]) & (wr > 0.5)
    wr *= _smoothstep((dn - P["neutral_black_d"]) * (1.0 / 0.1))
    if blk.any():
        rb = max(1, int(round(P["neutral_black_reach_in"] * dpi)))
        nearb = cv2.dilate(blk.view(np.uint8), np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)) > 0
        wr2 = cv2.subtract(1.0, _smoothstep((ratio - rlim) * (1.0 / P["neutral_r_w"])))
        wr2 *= nearb
        wr2 *= core
        np.maximum(wr, wr2, out=wr)
    np.maximum(wn, wr, out=wn)
    wn *= _smoothstep((dn - P["neutral_d_lo"]) * (1.0 / (P["neutral_d_hi"] - P["neutral_d_lo"])))      # near-paper pixels: nothing to neutralise
    r = max(1, int(round(P["neutral_reach_in"] * dpi)))
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    ncore = (core & (wn > 0.5)).view(np.uint8)
    if ncore.any():
        ccore = (core & (ratio > P["neutral_color"])).view(np.uint8)
        edge = (cv2.dilate(ncore, ker) > 0) & (dn < P["neutral_edge_d"]) & (ratio > P["neutral_edge_ratio"])
        if ccore.any():
            edge &= cv2.dilate(ccore, ker) == 0
        # broad coloured areas (highlighter) are not fringe: keep what survives a ~neutral_area_in opening of the coloured mask
        # (computed on a 1/3 grid; a fringe band is far thinner than a highlighter stroke)
        sc = 3
        cm = ((dn >= 0.03) & (ratio > 0.5)).view(np.uint8)
        cms = cv2.resize(cm, (max(1, cm.shape[1] // sc), max(1, cm.shape[0] // sc)), interpolation=cv2.INTER_AREA)
        ko = max(2, int(round(P["neutral_area_in"] * dpi / sc)))
        big = cv2.morphologyEx((cms > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
        if big.any():
            edge &= cv2.resize(big, (cm.shape[1], cm.shape[0]), interpolation=cv2.INTER_NEAREST) == 0
        # Only chromatic, near-paper-luminance pixels in long runs are treated as fringe: grey halftone, faint print and short bits beside
        # glyphs cannot be told from faint real ink (a softer tier that neutralised those cost 0.003 faint retention in bench/eval_clean).
        strict = edge & (cv2.add(cv2.add(qc[0], qc[1]), qc[2]) > 3.0 - 3.0 * P["neutral_edge_mean"])
        # a long thin band along a rule / underline (JPEG chroma subsampling, channel misregistration) is not ink: its density becomes min r
        L = max(3, int(round(P["neutral_run_in"] * dpi)))
        e8 = strict.view(np.uint8)
        run = (cv2.morphologyEx(e8, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) > 0) | (cv2.morphologyEx(e8, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8)) > 0)
        we = run.astype(np.float32)
        np.maximum(wn, we, out=wn)
        return wn, we
    return wn, None


def _open_runs(key, n: int, L: int):
    """Sorted unique keys `line * n + pos` of the set pixels of 1-D lines of length `n` -> the keys kept by an opening with a length-L
    kernel (anchor L // 2) of every line.  Equal to cv2.morphologyEx(MORPH_OPEN) with a (1, L) kernel, including its border rule (the
    erosion ignores pixels beyond the image; the opening of a run is shifted by one pixel for even L).  A run [a, b] survives the erosion
    at x in [xlo, xhi]: its window [x - L//2, x + L - 1 - L//2] clipped to the line lies in the run; the dilation spreads x to
    [x - (L - 1 - L//2), x + L//2]."""
    m = key.size
    if m == 0:
        return key
    a0 = L // 2
    brk = np.empty(m, bool)
    brk[0] = True
    np.not_equal(np.diff(key), 1, out=brk[1:])
    brk[1:] |= (key[1:] % n) == 0
    st = np.flatnonzero(brk)
    ln = np.diff(np.concatenate((st, [m])))
    k0 = key[st]
    line = k0 // n
    pa = k0 - line * n                                   # first / last position of every run
    pb = pa + ln - 1
    xlo = np.where(pa == 0, 0, pa + a0)
    xhi = np.where(pb == n - 1, n - 1, pb - (L - 1 - a0))
    ok = xlo <= xhi
    lo = np.maximum(xlo[ok] - (L - 1 - a0), 0)
    hi = np.minimum(xhi[ok] + a0, n - 1)
    cnt = hi - lo + 1
    if cnt.size == 0:
        return key[:0]
    base = line[ok] * n + lo
    off = np.cumsum(cnt) - cnt
    return np.repeat(base - off, cnt) + np.arange(int(cnt.sum()))


def _open_both(E, h: int, w: int, L: int):
    """Flat indices of the pixels kept by the (1, L) or the (L, 1) opening of the binary image with the set pixels `E` (see `_open_runs`)."""
    y = E // w
    x = E - y * w
    k1 = np.sort(E)
    r1 = k1                                              # row-major keys == flat indices
    r1 = _open_runs(r1, w, L)
    k2 = np.sort(x * h + y)                              # column-major keys
    r2 = _open_runs(k2, h, L)
    return r1, (r2 % h) * w + r2 // h


def _neutral_sparse(q, qmin, idx, h: int, w: int, dpi: float, P: dict, pcol):
    """`_neutral_weight` evaluated only at the flat pixel indices `idx` (the snap is applied nowhere else): (wn[idx], we[idx] or None),
    bit-identical to the whole-plane version.  Everything is pointwise in (q_c) except the dilations / openings, which run on uint8 planes
    scattered from sparse pixel sets (or, for the run openings, on the few lines that can have one): the weight is exactly 0 where the ink
    density is <= neutral_d_lo, so the main part only needs the pixels above it; the fringe part (long chroma-only bands beside neutral
    cores) looks at the dilated neutral cores and, for the coloured-area test, at the pixels with ink density >= 0.03."""
    qf = [c.ravel() for c in q]
    d_lo, d_hi = P["neutral_d_lo"], P["neutral_d_hi"]
    one, tiny = np.float32(1.0), np.float32(0.02)

    def planes(ii):                   # clipped transmittance at flat indices ii -> (dn, ratio, sum of the q_c, chrom)
        qc = [np.minimum(np.take(c, ii), one) for c in qf]
        qmn = np.minimum(np.minimum(qc[0], qc[1]), qc[2])
        chrom = np.maximum(np.maximum(qc[0], qc[1]), qc[2]) - qmn
        dn = one - qmn
        return dn, chrom, chrom / np.maximum(dn, tiny), qc[0] + qc[1] + qc[2]

    pc = float((pcol.max() - pcol.min()) / max(pcol.max(), 1.0))
    clim = np.float32(pc * P["neutral_c_k"] + P["neutral_c0"])
    rlim = np.float32(pc * P["neutral_r_k"] + P["neutral_r0"])
    A3 = _nonzero(cv2.compare(qmin, 1.0 - 0.03 + 1e-3, cv2.CMP_LT))      # superset of {ink density >= 0.03}
    dn3, chrom3, ratio3, sum3 = planes(A3)
    sel = dn3 > d_lo - 0.01                                               # superset of {ink density > neutral_d_lo}
    A, dn, chrom, ratio = A3[sel], dn3[sel], chrom3[sel], ratio3[sel]
    del chrom3
    wmain = np.zeros(h * w, np.float32)
    ncore_pl = ccore_pl = None
    if A.size:
        wn = one - _smoothstep((chrom - clim) * (1.0 / P["neutral_c_w"]))
        core = dn >= P["neutral_core_d"]
        wr = one - _smoothstep((ratio - rlim) * (1.0 / P["neutral_r_w"]))
        blk = (dn >= P["neutral_black_d"]) & (wr > 0.5)
        wr0 = wr.copy() if blk.any() else None
        wr *= _smoothstep((dn - P["neutral_black_d"]) * (1.0 / 0.1))
        if wr0 is not None:
            rb = max(1, int(round(P["neutral_black_reach_in"] * dpi)))
            bp = np.zeros(h * w, np.uint8)
            bp[A[blk]] = 1
            nearb = cv2.dilate(bp.reshape(h, w), np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)).ravel()[A]
            wr0 *= nearb
            wr0 *= core
            np.maximum(wr, wr0, out=wr)
        np.maximum(wn, wr, out=wn)
        wn *= _smoothstep((dn - d_lo) * (1.0 / (d_hi - d_lo)))
        wmain[A] = wn
        nc = core & (wn > 0.5)
        if nc.any():
            ncore_pl = np.zeros(h * w, np.uint8)
            ncore_pl[A[nc]] = 1
            cc = core & (ratio > P["neutral_color"])
            if cc.any():
                ccore_pl = np.zeros(h * w, np.uint8)
                ccore_pl[A[cc]] = 1
        del wn, wr, wr0, core, blk
    del dn, ratio, chrom
    wn_idx = wmain[idx]
    if ncore_pl is None:
        return wn_idx, None
    r = max(1, int(round(P["neutral_reach_in"] * dpi)))
    ker = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    dil = cv2.dilate(ncore_pl.reshape(h, w), ker)                         # near a neutral core: fringe candidates
    e_d, e_ratio, e_thr = P["neutral_edge_d"], P["neutral_edge_ratio"], 3.0 - 3.0 * P["neutral_edge_mean"]
    # candidates inside the ink-density >= 0.03 set (already evaluated) ...
    okA = (dil.ravel()[A3] > 0) & (dn3 < e_d) & (ratio3 > e_ratio) & (sum3 > e_thr)
    E1 = A3[okA]
    # ... and the remaining pixels of the dilated cores (ink density < 0.03)
    D2 = _nonzero(cv2.bitwise_and(dil, cv2.compare(qmin, 1.0 - 0.03 + 1e-3, cv2.CMP_GE)))
    dn2, _, ratio2, sum2 = planes(D2)
    E = np.concatenate((E1, D2[(dn2 < e_d) & (ratio2 > e_ratio) & (sum2 > e_thr)]))
    del dil, okA, D2, dn2, ratio2, sum2
    if ccore_pl is not None and E.size:
        E = E[cv2.dilate(ccore_pl.reshape(h, w), ker).ravel()[E] == 0]
    if E.size:
        # broad coloured areas (highlighter) are not fringe (computed on a 1/3 grid; a fringe band is far thinner than a highlighter stroke)
        sc = 3
        cm = np.zeros(h * w, np.uint8)
        cm[A3[(dn3 >= 0.03) & (ratio3 > 0.5)]] = 1
        cms = cv2.resize(cm.reshape(h, w), (max(1, w // sc), max(1, h // sc)), interpolation=cv2.INTER_AREA)
        ko = max(2, int(round(P["neutral_area_in"] * dpi / sc)))
        big = cv2.morphologyEx((cms > 0).astype(np.uint8), cv2.MORPH_OPEN, np.ones((ko, ko), np.uint8))
        if big.any():
            E = E[cv2.resize(big, (w, h), interpolation=cv2.INTER_NEAREST).ravel()[E] == 0]
    if not E.size:
        return wn_idx, None
    # Only chromatic, near-paper-luminance pixels in long runs are treated as fringe (see `_neutral_weight`)
    L = max(3, int(round(P["neutral_run_in"] * dpi)))
    if min(h, w) < 2 * L:
        e8 = np.zeros(h * w, np.uint8)
        e8[E] = 1
        e8 = e8.reshape(h, w)
        run = (cv2.morphologyEx(e8, cv2.MORPH_OPEN, np.ones((1, L), np.uint8)) > 0) | (cv2.morphologyEx(e8, cv2.MORPH_OPEN, np.ones((L, 1), np.uint8)) > 0)
        we = run.ravel()[idx].astype(np.float32)
    else:
        rp = np.zeros(h * w, np.uint8)
        for part in _open_both(E, h, w, L):
            rp[part] = 1
        we = rp[idx].astype(np.float32)
    return np.maximum(wn_idx, we), we


def _snap(rs, wn, we=None):
    """In place: r_c += wn * (target - r_c) with target = max(r) (the darkness is kept), min(r) on the fringe pixels `we` (a fringe is not ink)."""
    m = np.maximum(np.maximum(rs[0], rs[1]), rs[2])
    if we is not None:
        mn = np.minimum(np.minimum(rs[0], rs[1]), rs[2])
        mn -= m
        mn *= we
        m += mn
    for c in range(3):
        d = m - rs[c]
        d *= wn
        rs[c] += d


_DENSE_FRAC = 0.35     # above this fraction of candidate ink pixels the tone mapping runs on whole planes


def clean_page(rgb: np.ndarray, dpi: float, paper: str = "flatten", **params) -> np.ndarray:
    """Clean one page (uint8 HxWx3 RGB) at its native resolution `dpi`; returns uint8 HxWx3.

    Same maths as research/r2_cleaning/clean.py, restructured for speed:
      * works on transmittance q_c = rgb_c * (1/bg_c) (upsampled reciprocal background: one multiply per pixel)
        instead of r_c = 1 - rgb_c/bg_c, so d = 1 - min_c q_c and the chroma term max q - min q need no extra planes;
      * everything after the soft gate (faint boost, tone curve, uint8 conversion) is evaluated only where the gate can
        be non-zero (paper pixels are exactly white / the paper tint), via index arrays; dense pages use whole planes.
    """
    P = dict(DEFAULTS)
    P.update(params)
    if paper not in ("flatten", "keep"):
        raise ValueError("paper must be 'flatten' or 'keep'")
    h, w = rgb.shape[:2]
    if min(h, w) < 2 or P["gamma"] != 1.0:    # degenerate size (OpenCV treats 1x1 arrays as scalars) / unfused tone curve
        return clean_page_reference(rgb, dpi, paper=paper, **params)
    bgs = np.maximum(estimate_background(rgb, dpi, P["bg_win_in"], P["guard_win_in"], P["blob_guard"],
                                         P["bg_work_dpi"], P["bg_open_in"]), 16.0)
    paper_col = None
    if paper == "keep" or P["neutral_snap"]:      # global paper colour (median of the brighter half of the background map)
        paper_col = _paper_colour_bg(bgs)
    ibg = 1.0 / bgs
    # transmittance planes q_c = rgb_c / bg_c (NOT clipped at 1; paper ~ 1, ink < 1)
    q = []
    ht_s = P["desc_sigma_in"] > 0 and min(h, w) >= 32 and (P["ht_tone"] or P["pk_seed"] > 0 or P["pk_gain"] > 0)
    ups = []
    for c, plane in enumerate(cv2.split(rgb)):
        up = cv2.resize(np.ascontiguousarray(ibg[..., c]), (w, h), interpolation=cv2.INTER_LINEAR)
        q.append(cv2.multiply(plane, up, dtype=cv2.CV_32F))
        if ht_s:
            ups.append(up)
    del up
    rgb_sp = cv2.split(cv2.GaussianBlur(rgb, (0, 0), max(0.6, P["desc_sigma_in"] * dpi))) if ht_s else None   # descreened image

    def q_desc(ii):       # descreened transmittance planes q_c = blur(rgb)_c / bg_c at flat indices ii (None = everywhere)
        if ii is not None and ii.size < 5:         # OpenCV treats 1-D arrays of 1-4 elements as scalars (wrong output shape)
            return [rgb_sp[c].ravel()[ii].astype(np.float32) * ups[c].ravel()[ii] for c in range(3)]
        return [cv2.multiply(rgb_sp[c].ravel() if ii is None else rgb_sp[c].ravel()[ii],
                             ups[c].ravel() if ii is None else ups[c].ravel()[ii], dtype=cv2.CV_32F).ravel() for c in range(3)]
    qmin = cv2.min(cv2.min(q[0], q[1]), q[2])
    np.minimum(qmin, 1.0, out=qmin)                   # d = 1 - qmin  (>= 0)
    # bq = blur(qmin) = 1 - dm
    bq = cv2.GaussianBlur(qmin, (0, 0), max(0.5, P["denoise_in"] * dpi)) if P["denoise_in"] > 0 else qmin
    t_lo, t_hi, t_unsup = P["t_lo"], P["t_hi"], P["t_hi_unsup"]
    if P["noise_k"] > 0:     # page-adaptive noise floor
        sub = 1.0 - bq[::4, ::4]
        pp = sub[sub < 0.12]
        if pp.size > 1000:
            f = float(np.median(pp))
            sig = 1.4826 * float(np.median(np.abs(pp - f)))
            floor = f + P["noise_k"] * sig
            if floor > t_lo:
                t_hi += floor - t_lo
                t_unsup = max(t_unsup, floor + (P["t_hi_unsup"] - P["t_lo"]))
                t_lo = floor
    # x = u - t_lo with u = dm + E * sharp_w/8 (E = |grad blur(d)|)
    if P["sharp_w"] > 0:
        g1 = cv2.GaussianBlur(qmin, (0, 0), max(0.6, P["sharp_sigma_in"] * dpi))
        gx = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
        del g1
        E = cv2.magnitude(gx, gy)
        del gx, gy
        x = cv2.addWeighted(E, P["sharp_w"] / 8.0, bq, -1.0, 1.0 - t_lo)
        del E
    else:
        x = cv2.subtract(1.0 - t_lo, bq)
    wl = _light_lowres(bq, h, w, dpi, P, 4) if ht_s and (P["ht_tone"] or P["pk_scope"] == "light") else None
    # hysteresis: weak isolated components (bleed-through, speckle) need the higher threshold
    seed_thr = 1.0 - P["t_seed"]
    mask = cv2.compare(bq, 1.0 - t_lo, cv2.CMP_LT)
    nl, lab = cv2.connectedComponents(mask, connectivity=8)
    labf = lab.ravel()
    has = np.zeros(nl, bool)
    seeds = _nonzero(cv2.compare(bq, seed_thr, cv2.CMP_LT))      # seeds are a subset of the mask
    has[labf[seeds]] = True
    mi = _nonzero(mask)
    Lmi = labf[mi]
    has[0] = False
    added = None
    if P["comp_min_in2"] > 0:     # large components are supported on their own (areas from the mask pixels only)
        amin = P["comp_min_in2"] * dpi * dpi
        has |= np.bincount(Lmi, minlength=nl) > amin
        has[0] = (labf.size - mi.size) > amin          # background label counts like any other (as in the reference)
    if P["line_in"] > 0:
        has, added = _line_support(has, None, None, labf, mi, 1.0 - bq.ravel()[mi], 1.0 - qmin.ravel()[mi], h, w, dpi, P, Lmi=Lmi)
    if ht_s and (P["pk_seed"] > 0 or P["pk_gain"] > 0):
        # thin / halftoned light strokes, evaluated only on mask pixels of still-unsupported components: a descreened core
        # >= pk_seed supports its component (bleed never gets that dark) and the gate density is at least pk_gain * descreened
        # density there.  Components that are already supported (dark ink, large) are untouched.
        unsup = ~has[Lmi]
        mu = mi[unsup]
        if P["pk_scope"] != "unsup":
            if P["pk_scope"] == "all":
                gm = mi
            else:                                  # unsupported components + light-ink regions
                gm = mi[unsup | (_sample_lowres(wl, mi, w, 4) > 0.01)]
        else:
            gm = mu
        if gm.size:
            qd = q_desc(gm)
            dpk = 1.0 - np.minimum(np.minimum(qd[0], qd[1]), qd[2])
            if P["pk_seed"] > 0:
                if gm is mu:
                    has[labf[mu[dpk > P["pk_seed"]]]] = True
                else:
                    has_u = ~has[labf[gm]] & (dpk > P["pk_seed"])      # seeds only for components that were unsupported
                    has[labf[gm[has_u]]] = True
            if P["pk_gain"] > 0:
                xf0 = x.ravel()
                xf0[gm] = np.maximum(xf0[gm], P["pk_gain"] * dpk - t_lo)
            del qd, dpk
        del mu
    del mask, mi, Lmi
    pen = np.where(has, 0.0, t_unsup - t_lo).astype(np.float32)      # per-component gate penalty
    if added is not None:       # line-supported faint words: gate opened wider (their density is low by construction)
        pen[added] = -P["line_open"] * (t_hi - t_lo)
    inv_span = 1.0 / max(t_hi - t_lo, 1e-3)

    # ---- tone mapping only where the gate is open (x > 0 before the penalty; the penalty only lowers x)
    xf = x.ravel()
    idx = _nonzero(cv2.compare(x, 0.0, cv2.CMP_GT))
    dense = idx.size > _DENSE_FRAC * xf.size
    if dense:
        idx = None
        xs = xf - pen[labf]
    else:
        xs = xf[idx] - pen[labf[idx]]
        sel = np.flatnonzero(xs > 0)                # pixels whose penalised gate is still open
        idx, xs = idx[sel], xs[sel]
        if 0 < idx.size < 5:       # OpenCV treats 1-D arrays of 1-4 elements as scalars: repeated pixels (same value written twice) avoid that
            idx, xs = np.resize(idx, 5), np.resize(xs, 5)
    del x, lab, labf
    wht = None
    if wl is not None and P["ht_tone"]:    # weight of the descreened tone: 1 in regions with only light ink
        wht = _up_nearest(wl, h, w, 4).ravel() if idx is None else _sample_lowres(wl, idx, w, 4)
        if idx is None:
            wht[wht <= 0.01] = 0.0
    xs *= inv_span
    alpha = _smoothstep(xs)
    wnp = wep = None
    if P["neutral_snap"]:
        if idx is None or min(h, w) < 2:
            wnp, wep = _neutral_weight(q, dpi, P, bgs, paper_col)
        else:                                            # weights at the open-gate pixels only (the snap is applied nowhere else)
            wnp, wep = _neutral_sparse(q, qmin, idx, h, w, dpi, P, paper_col)
    del qmin
    if idx is None:
        qs = [qc.ravel() for qc in q]
        dms = 1.0 - bq.ravel()
    else:
        qs = [qc.ravel()[idx] for qc in q]
        dms = 1.0 - bq.ravel()[idx]
    del q, bq
    # r_c = ink density = max(0, 1 - q_c)
    rs = []
    for qc in qs:
        r = np.subtract(np.float32(1.0), qc)
        np.maximum(r, 0.0, out=r)
        rs.append(r)
    del qs
    if wht is not None:      # blend raw and descreened ink density by wht
        sub = None if idx is None else np.flatnonzero(wht > 0.01)      # only where the descreened tone has weight
        if sub is None or sub.size:
            qdl = q_desc(idx if sub is None else idx[sub])
            wsub = wht if sub is None else wht[sub]
            for c in range(3):
                rd = np.subtract(np.float32(1.0), qdl[c])
                np.maximum(rd, 0.0, out=rd)
                if sub is None:
                    rd -= rs[c]
                    rd *= wsub
                    rs[c] += rd
                else:
                    rc = rs[c][sub]
                    rd -= rc
                    rd *= wsub
                    rs[c][sub] = rc + rd
    if wnp is not None:
        if idx is None:
            wnp = wnp.ravel()
            wep = None if wep is None else wep.ravel()
        _snap(rs, wnp, wep)
    del wnp, wep
    chroma = np.maximum(np.maximum(rs[0], rs[1]), rs[2]) - np.minimum(np.minimum(rs[0], rs[1]), rs[2])
    # a = alpha * (1 / (1 - bp)) * k   (k = hue-preserving boost of faint grey ink)
    a = alpha * np.float32(1.0 / (1.0 - P["bp"]))
    if P["faint_gain"] != 1.0:
        fd = _smoothstep((dms - P["faint_lo"]) * (1.0 / (P["faint_hi"] - P["faint_lo"])))
        fc = _smoothstep((chroma - P["faint_chroma_lo"]) * (1.0 / (P["faint_chroma_hi"] - P["faint_chroma_lo"])))
        fd -= 1.0                                    # -(1 - fd)
        fd *= (1.0 - fc)
        fd *= -(P["faint_gain"] - 1.0)               # k - 1 = (gain - 1) (1 - fd)(1 - fc)
        fd += 1.0
        a *= fd
    h, w = rgb.shape[:2]
    res = np.empty((h, w, 3), np.uint8)
    scales = [255.0 * (paper_col[c] / 255.0 if paper == "keep" else 1.0) for c in range(3)]
    fill = np.array([min(max(sc + 0.5, 0.0), 255.0) for sc in scales], np.uint8)     # alpha == 0 -> white / paper tint
    res[:] = np.tile(fill, (w, 1))                  # row broadcast (a 3-vector broadcast over the page is 10x slower)
    vals = np.empty((rs[0].size, 3), np.uint8)
    for c in range(3):
        t = rs[c]
        t *= a
        np.minimum(t, alpha, out=t)                  # dk * alpha with dk = min(r * inv * k, 1)
        t *= -scales[c]
        t += scales[c] + 0.5
        vals[:, c] = t.astype(np.uint8)
    if idx is None:
        res.reshape(-1, 3)[:] = vals
    else:
        res.reshape(-1, 3)[idx] = vals
    return res


def clean_page_reference(rgb: np.ndarray, dpi: float, paper: str = "flatten", **params) -> np.ndarray:
    """Straightforward (slow) implementation: used for degenerate tiny inputs and as the reference in tests/bench.
    `clean_page` is the optimised equivalent (max abs diff ~1).  With desc_sigma_in=0 (halftone handling off) it is identical
    to research/r2_cleaning/clean.py."""
    P = dict(DEFAULTS)
    P.update(params)
    if paper not in ("flatten", "keep"):
        raise ValueError("paper must be 'flatten' or 'keep'")
    h, w = rgb.shape[:2]
    bgs = np.maximum(estimate_background(rgb, dpi, P["bg_win_in"], P["guard_win_in"], P["blob_guard"],
                                         P["bg_work_dpi"], P["bg_open_in"]), 16.0)
    lumbg = bgs.mean(-1)
    paper_col = np.median(bgs[lumbg >= np.percentile(lumbg, 50)], axis=0)   # global paper colour (brighter half)
    # planar float32 ink density r_c = max(0, 1 - rgb_c / bg_c)
    r = []
    for c in range(3):
        bgc = cv2.resize(np.ascontiguousarray(bgs[..., c]), (w, h), interpolation=cv2.INTER_LINEAR)
        rc = cv2.subtract(bgc, rgb[..., c].astype(np.float32))
        cv2.divide(rc, bgc, rc)
        np.maximum(rc, 0, out=rc)
        r.append(rc)
    ht = P["desc_sigma_in"] > 0 and min(h, w) >= 32 and (P["ht_tone"] or P["pk_seed"] > 0 or P["pk_gain"] > 0)
    if ht:        # descreened ink density planes (blur the 8-bit image, then the same flat-field division)
        rgb_s = cv2.GaussianBlur(rgb, (0, 0), max(0.6, P["desc_sigma_in"] * dpi))
        rsm = []
        for c in range(3):
            ibgc = cv2.resize(np.ascontiguousarray(1.0 / bgs[..., c]), (w, h), interpolation=cv2.INTER_LINEAR)
            rsm.append(np.maximum(0, 1.0 - rgb_s[..., c].astype(np.float32) * ibgc))
        ds = cv2.max(cv2.max(rsm[0], rsm[1]), rsm[2])
    r0 = list(r)
    d = cv2.max(cv2.max(r[0], r[1]), r[2])
    rmin = cv2.min(cv2.min(r[0], r[1]), r[2])
    dm = cv2.GaussianBlur(d, (0, 0), max(0.5, P["denoise_in"] * dpi)) if P["denoise_in"] > 0 else d
    t_lo, t_hi, t_unsup = P["t_lo"], P["t_hi"], P["t_hi_unsup"]
    if P["noise_k"] > 0:     # page-adaptive noise floor
        sub = dm[::4, ::4]
        pp = sub[sub < 0.12]
        if pp.size > 1000:
            f = float(np.median(pp))
            sig = 1.4826 * float(np.median(np.abs(pp - f)))
            floor = f + P["noise_k"] * sig
            if floor > t_lo:
                t_hi += floor - t_lo
                t_unsup = max(t_unsup, floor + (P["t_hi_unsup"] - P["t_lo"]))
                t_lo = floor
    u = dm
    if P["sharp_w"] > 0:
        g1 = cv2.GaussianBlur(d, (0, 0), max(0.6, P["sharp_sigma_in"] * dpi))
        gx = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
        E = cv2.magnitude(gx, gy)
        u = dm + E * (P["sharp_w"] / 8.0)
    # hysteresis: weak isolated components (bleed-through, speckle) need the higher threshold
    inmask = dm > t_lo
    added = None
    nl, lab, cstats, ccent = cv2.connectedComponentsWithStats(inmask.astype(np.uint8), connectivity=8)
    has = np.zeros(nl, bool)
    has[np.unique(lab[dm > P["t_seed"]])] = True
    has[0] = False
    if P["comp_min_in2"] > 0:
        has |= (np.bincount(lab.ravel(), minlength=nl) > P["comp_min_in2"] * dpi * dpi)
    if P["line_in"] > 0:
        mi_r = np.flatnonzero(inmask.ravel())
        has, added = _line_support(has, cstats, ccent, lab.ravel(), mi_r, dm.ravel()[mi_r], d.ravel()[mi_r], h, w, dpi, P)
    wm = None
    if ht and (P["ht_tone"] or P["pk_scope"] == "light"):     # light-region weight (see _light_lowres)
        wm = _up_nearest(_light_lowres(1.0 - dm, h, w, dpi, P, 4), h, w, 4)
        wm[wm <= 0.01] = 0.0
    if ht and (P["pk_seed"] > 0 or P["pk_gain"] > 0):     # only components still unsupported (see clean_page)
        unsup_px = inmask & ~has[lab]
        has_base = has.copy()
        if P["pk_seed"] > 0:
            has[np.unique(lab[unsup_px & (ds > P["pk_seed"])])] = True
    sup = has[lab]
    x = u - t_lo
    if ht and P["pk_gain"] > 0:
        gm = {"unsup": unsup_px, "all": inmask, "light": unsup_px | (inmask & (wm > 0.01) if wm is not None else False)}[P["pk_scope"]]
        x = np.where(gm, np.maximum(x, P["pk_gain"] * ds - t_lo), x).astype(np.float32)
    pen_r = np.where(sup, 0.0, t_unsup - t_lo).astype(np.float32)
    if added is not None:
        pen_r[added[lab]] = -P["line_open"] * (t_hi - t_lo)
    x -= pen_r
    x *= 1.0 / max(t_hi - t_lo, 1e-3)
    alpha = _smoothstep(x)
    d_chroma = d - rmin
    if ht and P["ht_tone"]:     # blend raw and descreened ink density by the light-region weight
        r = [rc + wm * (rd - rc) for rc, rd in zip(r, rsm)]
        d_chroma = cv2.max(cv2.max(r[0], r[1]), r[2]) - cv2.min(cv2.min(r[0], r[1]), r[2])
    if P["neutral_snap"]:
        wn, we = _neutral_weight([1.0 - rc for rc in r0], dpi, P, bgs)
        _snap(r, wn, we)
        d_chroma = cv2.max(cv2.max(r[0], r[1]), r[2]) - cv2.min(cv2.min(r[0], r[1]), r[2])
    k = None
    if P["faint_gain"] != 1.0:   # hue-preserving boost of faint grey ink
        fd = _smoothstep((dm - P["faint_lo"]) * (1.0 / (P["faint_hi"] - P["faint_lo"])))
        fc = _smoothstep((d_chroma - P["faint_chroma_lo"]) * (1.0 / (P["faint_chroma_hi"] - P["faint_chroma_lo"])))
        k = 1.0 + (P["faint_gain"] - 1.0) * (1.0 - fd) * (1.0 - fc)
    inv = 1.0 / (1.0 - P["bp"])
    out = []
    for rc in r:
        dk = rc * inv
        if k is not None:
            dk *= k
        np.minimum(dk, 1.0, out=dk)
        if P["gamma"] != 1.0:
            dk = 1.0 - (1.0 - dk) ** P["gamma"]
        dk *= alpha
        out.append(1.0 - dk)
    res = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        o = out[c] * (255.0 * (paper_col[c] / 255.0 if paper == "keep" else 1.0))
        o += 0.5
        res[..., c] = np.clip(o, 0, 255).astype(np.uint8)
    return res
