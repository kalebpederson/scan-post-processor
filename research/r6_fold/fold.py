"""Fold-crease detection and removal (R6 prototype, written to be ported into scanpp).

API (all lengths in inches, dpi-independent; coordinates of a Fold are fractions / angles so they apply at any resolution):

    folds = detect_folds(img, dpi, P=FoldParams())   # img: uint8 gray HxW or RGB HxWx3 at any dpi; ~35 ms at 8 Mpx
    mask  = fold_mask(rgb, dpi, folds, P)            # uint8 HxW at the resolution of `rgb`; 1 = crease pixel (ink crossing it excluded)
    out   = remove_folds(rgb, dpi, folds, P)         # rgb with the crease pixels replaced by local paper (new array)

Pipeline placement (see REPORT.md): detect on the edge-cleaned work-resolution page, remove on the native RGB *before* clean_page
and before blank_score (so the line support of the cleaner and the blank score never see the crease).

What a "fold" is here: a near-horizontal (|angle| <= ang_max) faint ridge of ink density that runs across the whole sheet, i.e. is
present (relative to its own median strength) in the outermost windows on both sides, excluding strong ink pixels (text crossing it
leaves it unobserved there, not absent); that is not periodic (ruled / graph paper); is not a dark rule (amp <= amp_max) and
not a paper-edge step.  Vertical folds are not handled (a fold is parallel to a page edge: rotate the page and call again).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class FoldParams:
    # --- detection (work resolution) ---
    work_dpi: float = 100.0         # detection resolution (images above are area-downscaled to ~this)
    ang_max: float = 3.0            # deg, search range of the line angle
    ang_step: float = 0.25
    strong: float = 0.5             # ink density above which a pixel is "ink" (excluded from the line statistics) ...
    strong_dil_in: float = 0.02     # ... together with this dilation
    chroma_d: float = 0.30          # coloured-ink test: density >= chroma_d and (max q - min q) >= chroma_ratio * density (0.12 flagged the bluish crease itself as ink)
    chroma_ratio: float = 0.4       # (creases: median 0.13, p90 0.3; pen ink: median 0.4-0.5, p90 0.9)
    cont_t: float = 0.01            # continuity: a column shows the line when its 5-row max exceeds the side rows by this
    cap: float = 0.3                # ink density cap of the remaining pixels
    fast: bool = True               # stage 1 without image warps (shifted sub-window sums); False = exact shear
    sub_win_px: float = 12.0        # width of the sub-windows of the fast path (at the work resolution)
    x_margin: float = 0.03          # the profile starts this fraction of the page width from each side
    cand_min: float = 0.2           # stage-1 candidates: score above this
    sep_in: float = 0.15            # non-maximum suppression distance between lines
    n_cand: int = 30
    win_in: float = 0.25            # window length of the along-line presence profile
    cov_base: float = 0.85          # ... and this fraction of the rows just beside the line (no text next to it)
    cov_obs: float = 0.5            # a window is "observed" when this fraction of its line pixels is not ink
    z_min: float = 4.0              # presence: ridge / robust noise of the window
    rel_pres: float = 0.35          # presence: ridge >= rel_pres * median ridge of the line
    pres_frac_min: float = 0.75     # fraction of observed windows that must show the line
    obs_win_min: int = 6            # observed windows: at least max(obs_win_min, obs_frac_min * windows)
    obs_frac_min: float = 0.4
    cont_min: float = 0.55          # mean continuity over the observed windows (columns that show the line)
    crowd_in: float = 0.1           # (fraction of the page height) another strong line this close makes both rules / table / grid lines
    crowd_score: float = 0.8
    rule_dark: float = 0.55         # native check: a column is a dark rule when the density along the line is >= this ...
    rule_iso_d: float = 0.3         # ... and the rows 0.03..0.05 in beside it are free of ink (density < this)
    rule_frac_max: float = 0.25     # a line whose columns are dark rule for more than this fraction is a printed / drawn rule, not a crease
    use_edge: bool = True           # ablation switch of the edge criterion
    edge_rel_min: float = 0.3       # ridge in the outermost window on each side relative to the median ridge
    step_ratio: float = 1.0         # ridge must be >= step_ratio * |mean above - mean below|
    amp_min: float = 0.02           # ridge amplitude (mean capped ink density over the line, above its baseline)
    amp_max: float = 0.14           # darker lines are printed / drawn rules, not creases
    cv_mean_min: float = 0.75       # mean fraction of non-ink pixels along the line (a printed rule is ink along its whole length)
    margin_frac: float = 0.02       # ignore lines this close to the top / bottom edge of the page (edge shadow)
    min_page_in: tuple = (5.0, 7.0) # (width, height) in inches: smaller images are cut-outs, not folded sheets
    prior_pos: tuple = (1 / 3, 1 / 2, 2 / 3)   # customary fold positions (letter tri-fold, half fold), fractions of the page height
    prior_tol: float = 0.02
    pair_score_min: float = 0.6     # tier B: own score needed ...
    pair_partner_min: float = 0.8   # ... when the partner side has a line of at least this score
    pair_tol: float = 0.015         # tier B: same relative row within this
    family_score_min: float = 0.6   # tier F: score needed for a line 1/3 of the page height from a tier-A fold (tri-fold family)
    family_spacing: float = 1 / 3
    family_tol: float = 0.015
    # --- trajectory / mask (native resolution) ---
    traj_half_in: float = 0.07      # search half-width around the Radon line
    traj_block_in: float = 0.0133   # column block of the tracker
    traj_step: int = 2              # max row change per block (in blocks of 0.0133 in: 2 px at 300 ppi)
    band_in: float = 0.017          # crease half-width around the trajectory (0.017 in = 5 px at 300 ppi)
    tone_dark: float = 0.40         # ink density >= this is never removed (real ink; crease pixels are < ~0.40 even at 300 ppi, pencil is not)
    stroke_t: float = 0.12          # a vertical run of density >= stroke_t crossing the band edge marks a stroke (protected)
    stroke_reach_in: float = 0.013  # ... when it reaches this far beyond the band
    protect_dil_in: float = 0.0033  # protection spreads sideways by this much
    orient_h_min: float = 0.7       # a pixel is crease only where the local ink structure is horizontal: Jyy / (Jxx + Jyy) >= this (0 disables)
    orient_sigma_in: float = 0.01   # structure tensor window
    orient_e0: float = 1e-4         # below this tensor energy there is no structure (faint paper texture): crease-compatible
    protect_chroma: bool = True     # coloured-ink pixels are protected like dark ink
    halo_in: float = 0.008          # pixels this close to dark ink (density >= tone_dark) are ink skirt, not crease


@dataclass
class Fold:
    y_frac: float                   # row of the line at the page centre / page height
    angle_deg: float                # tilt: y(x) = y0 + tan(angle) * (x - W/2)
    amp: float
    pres_frac: float
    edge_rel: tuple                 # (left, right) ridge in the outermost windows relative to the median
    n_obs: int
    info: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ helpers
def _to_gray(img):
    if img.ndim == 3:
        return img.mean(2, dtype=np.float32) if img.dtype != np.float32 else img.mean(2)
    return img.astype(np.float32)


def _bg(g, dpi, bg_work=50.0, win_in=0.35, open_in=0.06):
    """Paper background (grey opening -> local max), same idea as scanpp.clean.estimate_background, single channel."""
    h, w = g.shape
    s = max(1.0, dpi / bg_work)
    small = cv2.resize(g, (max(8, int(round(w / s))), max(8, int(round(h / s)))), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(win_in * bg_work)) | 1)
    ko = max(2, int(round(open_in * bg_work)))
    small = cv2.morphologyEx(small, cv2.MORPH_OPEN, np.ones((ko, ko), np.float32))
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.GaussianBlur(bg, (0, 0), k / 2.5)


def ink_density(g, dpi):
    """1 - g / local paper, clipped to [0, 1]."""
    bg = cv2.resize(_bg(g, dpi), (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(1.0 - g / np.maximum(bg, 16.0), 0.0, 1.0)


def _shear(a, t):
    h, w = a.shape
    M = np.array([[1, 0, 0], [t, 1, -t * w / 2]], np.float32)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _work_img(img, dpi, P):
    """uint8 gray HxW or RGB HxWx3 at any dpi -> float32 image at ~work_dpi (area-downscaled when above)."""
    a = img
    d = dpi
    if dpi > P.work_dpi * 1.05:
        s = P.work_dpi / dpi
        a = cv2.resize(img, (max(8, int(round(img.shape[1] * s))), max(8, int(round(img.shape[0] * s)))), interpolation=cv2.INTER_AREA)
        d = P.work_dpi
    return a.astype(np.float32), d


def density_maps(img, dpi, P=FoldParams(), native=False):
    """(ink density 1 - min_c rgb_c / bg_c, coloured-ink mask, work dpi).  The flat-field division is per channel (coloured paper), so a
    crease (a shadow: all channels scale alike) is neutral in q while pen ink changes the channel ratios: pixels with density >= chroma_d
    and chroma / density >= chroma_ratio are coloured ink (not crease)."""
    a, d = (img.astype(np.float32), dpi) if native else _work_img(img, dpi, P)
    if a.ndim == 2:
        a = a[..., None]
    q = []
    for c in range(a.shape[2]):
        bg = cv2.resize(_bg(np.ascontiguousarray(a[..., c]), d), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        q.append(np.minimum(a[..., c] / np.maximum(bg, 16.0), 1.0))
    qmin = q[0]
    qmax = q[0]
    for c in q[1:]:
        qmin = np.minimum(qmin, c)
        qmax = np.maximum(qmax, c)
    dens = 1.0 - qmin
    ink = (dens >= P.chroma_d) & ((qmax - qmin) >= P.chroma_ratio * dens)
    return dens, ink, d


# ------------------------------------------------------------------------------------------------ detection
def _evidence(dv, v, angs, dpi, P):
    """Dense evidence for every (angle, row): the sheared page is cut into windows of win_in along x; per window the ridge of the line at that
    row (max of 5 rows minus the mean of the rows 5..9 above and below, valid pixels only) is compared to the window noise.  Returns, for the
    best angle of every row: score, angle index and the statistics (dict of (H,) arrays)."""
    h, w = dv.shape
    x0 = int(w * P.x_margin)
    ww = max(4, int(P.win_in * dpi))
    nw = (w - 2 * x0) // ww
    x1 = x0 + nw * ww
    ku = np.zeros((19, 1), np.float32)
    ku[:5] = 1 / 5.0                    # rows y-9 .. y-5
    kd = ku[::-1].copy()                # rows y+5 .. y+9
    k5 = np.ones((5, 1), np.uint8)
    best = np.full(h, -1e9, np.float32)
    bi = np.zeros(h, np.int32)
    keep = {}
    n_min = P.obs_win_min
    if P.fast:
        # no image warps: the column sums of narrow sub-windows (sw columns) are computed once; a tilted line crosses sub-window j at the row offset
        # t * (x_j - w / 2), so the window profile of angle `a` is the sub-window sums shifted by those integer offsets (the line's tilt inside one sub-window,
        # <= tan(ang_max) * sw rows, is ignored).
        sw = ww // max(1, int(round(ww / P.sub_win_px))) if P.sub_win_px > 0 else ww
        sw = max(2, sw)
        while ww % sw:
            sw -= 1
        ns = nw * (ww // sw)
        sub_num = dv[:, x0:x1].reshape(h, ns, sw).sum(2)
        sub_den = v[:, x0:x1].reshape(h, ns, sw).sum(2)
        xc = x0 + (np.arange(ns) + 0.5) * sw
        yy = np.arange(h)[:, None]
    for ai, a in enumerate(angs):
        t = np.tan(np.radians(a))
        if P.fast:
            rows_ = np.clip(yy + np.round(t * (xc - w / 2.0)).astype(np.int64)[None, :], 0, h - 1)
            num = np.take_along_axis(sub_num, rows_, 0).reshape(h, nw, ww // sw).sum(2)
            den = np.take_along_axis(sub_den, rows_, 0).reshape(h, nw, ww // sw).sum(2)
        else:
            num = _shear(dv, t)[:, x0:x1].reshape(h, nw, ww).sum(2)
            den = _shear(v, t)[:, x0:x1].reshape(h, nw, ww).sum(2)
        prof = num / np.maximum(den, 1.0)
        cov = den * (1.0 / ww)
        up = cv2.filter2D(prof, -1, ku, borderType=cv2.BORDER_REPLICATE)
        dn = cv2.filter2D(prof, -1, kd, borderType=cv2.BORDER_REPLICATE)
        rid = cv2.dilate(prof, k5) - 0.5 * (up + dn)
        cl = cv2.blur(cov, (1, 5))
        cvb = np.minimum(cv2.filter2D(cov, -1, ku, borderType=cv2.BORDER_REPLICATE), cv2.filter2D(cov, -1, kd, borderType=cv2.BORDER_REPLICATE))
        sm = cv2.blur(prof, (1, 3))
        nz = 1.4826 * np.median(np.abs(sm - np.median(sm, 0)), 0) + 1e-4
        obs = (cl >= P.cov_obs) & (cvb >= P.cov_base)
        nobs = obs.sum(1)
        den_o = np.maximum(nobs, 1)
        mrid = (rid * obs).sum(1) / den_o
        rel = rid / np.maximum(mrid, 1e-6)[:, None]
        pres = ((rid / nz) > P.z_min) & (rel >= P.rel_pres) & obs
        pf = pres.sum(1) / den_o
        rel0 = np.where(obs[:, 0], rel[:, 0], 0.0)
        rel1 = np.where(obs[:, -1], rel[:, -1], 0.0)
        cvm = cl.mean(1)
        step = np.abs(((up - dn) * obs).sum(1)) / den_o
        edge_t = np.minimum(rel0, rel1) / P.edge_rel_min if P.use_edge else np.full(h, 9.0, np.float32)
        sc = np.minimum(np.minimum(pf / P.pres_frac_min, edge_t), cvm / P.cv_mean_min)
        valid = (nobs >= max(n_min, P.obs_frac_min * nw)) & (mrid >= P.amp_min) & (mrid <= P.amp_max) & (mrid >= P.step_ratio * step)
        sc = np.where(valid, sc, 0.0).astype(np.float32)
        sel = sc > best
        if sel.any():
            best[sel] = sc[sel]
            bi[sel] = ai
            for k, arr in (('pres', pf), ('rel0', rel0), ('rel1', rel1), ('cv_mean', cvm), ('amp', mrid), ('n_obs', nobs), ('step', step)):
                if k not in keep:
                    keep[k] = np.zeros(h, np.float32)
                keep[k][sel] = arr[sel]
    return best, bi, keep


def line_profile(img, dpi, y_frac, angle_deg, P=FoldParams(), _maps=None):
    """Per-window arrays of the line at (y_frac, angle) -- for analysis: dict(rid, z, cov, cvb, obs, pres, up, dn, ridc) where ridc is the
    per-column continuity (fraction of columns of the window whose 5-row maximum exceeds the mean of the rows 5..9 beside it by cont_t)."""
    if _maps is None:
        dens, cink, d = density_maps(img, dpi, P)
        S = ((dens > P.strong) | cink).astype(np.uint8)
        r = max(1, int(round(P.strong_dil_in * d)))
        v = 1.0 - cv2.dilate(S, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(np.float32)
        dv = np.minimum(dens, P.cap) * v
    else:
        dens, d = img, dpi
        dv, v = _maps
    h, w = dens.shape
    x0 = int(w * P.x_margin)
    ww = max(4, int(P.win_in * d))
    nw = (w - 2 * x0) // ww
    x1 = x0 + nw * ww
    t = np.tan(np.radians(angle_deg))
    sh, sv = _shear(dv, t)[:, x0:x1], _shear(v, t)[:, x0:x1]
    y = int(round(y_frac * h))
    num = sh.reshape(h, nw, ww).sum(2)
    den = sv.reshape(h, nw, ww).sum(2)
    prof = num / np.maximum(den, 1.0)
    up, dn = prof[y - 9:y - 4].mean(0), prof[y + 5:y + 10].mean(0)
    rid = prof[y - 2:y + 3].max(0) - 0.5 * (up + dn)
    cov = den / ww
    cl = cov[y - 2:y + 3].mean(0)
    cvb = np.minimum(cov[y - 9:y - 4].mean(0), cov[y + 5:y + 10].mean(0))
    sm = cv2.blur(prof, (1, 3))
    nz = 1.4826 * np.median(np.abs(sm - np.median(sm, 0)), 0) + 1e-4
    obs = (cl >= P.cov_obs) & (cvb >= P.cov_base)
    # per-column continuity
    colmax = sh[y - 2:y + 3].max(0)
    colbase = 0.5 * (sh[y - 9:y - 4].mean(0) + sh[y + 5:y + 10].mean(0))
    cont = ((colmax - colbase) > P.cont_t).astype(np.float32).reshape(nw, ww).mean(1)
    return dict(rid=rid, z=rid / nz, cov=cl, cvb=cvb, obs=obs, up=up, dn=dn, cont=cont, nz=nz)


def line_candidates(img, dpi, P=FoldParams()):
    """Stage 1: the lines of the page with a non-zero fold score (NMS over sep_in), best first.  Returns (list of dicts, (width_in, height_in)).
    A dict has y_frac, angle, amp (mean ridge over the observed windows), stats (pres, rel0, rel1, cv_mean, n_obs, step), score (>= 1: all criteria met)."""
    dens, cink, d = density_maps(img, dpi, P)
    h, w = dens.shape
    S = ((dens > P.strong) | cink).astype(np.uint8)
    r = max(1, int(round(P.strong_dil_in * d)))
    v = 1.0 - cv2.dilate(S, np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(np.float32)
    dv = np.minimum(dens, P.cap) * v
    angs = np.arange(-P.ang_max, P.ang_max + 1e-6, P.ang_step)
    best, bi, keep = _evidence(dv, v, angs, d, P)
    m = int(P.margin_frac * h)
    best = best.copy()
    best[:m] = 0
    best[h - m:] = 0
    out = []
    sep = int(P.sep_in * d)
    key = np.minimum(best, 3.0) + 1e-3 * keep['amp']      # score first, ridge amplitude as the tie-break (matters only for ablations with criteria switched off)
    for i in np.argsort(-key):
        if best[i] < P.cand_min or len(out) >= P.n_cand:
            break
        if all(abs(i - c['y_row']) > sep for c in out):
            st = {k: float(a[i]) for k, a in keep.items()}
            out.append(dict(y_row=int(i), y_frac=i / h, angle=float(angs[bi[i]]), amp=st['amp'], score=float(best[i]), stats=st))
    # continuity (per-column) only for the lines that pass everything else
    for c in out:
        if c['score'] >= P.crowd_score:
            lp = line_profile(dens, d, c['y_frac'], c['angle'], P, _maps=(dv, v))
            o = lp['obs']
            c['stats']['cont'] = float(lp['cont'][o].mean()) if o.any() else 0.0
            c['score'] = min(c['score'], c['stats']['cont'] / P.cont_min)
        else:
            c['stats']['cont'] = -1.0
    return out, (w / d, h / d)


def fold_score(c, P=FoldParams()):
    """Decision statistic of a candidate: min over the criteria of (value / threshold); >= 1 passes all of them."""
    return c['score']


def _near_prior(y, P):
    return min(abs(y - p) for p in P.prior_pos) <= P.prior_tol


def _uncrowded(cands, P):
    """Drop the lines that have another strong line (score >= crowd_score) within crowd_in of the page height: rulings, tables, grids."""
    strong = [c for c in cands if c['score'] >= P.crowd_score]
    out = []
    for c in cands:
        if c['score'] >= P.crowd_score and any(o is not c and abs(o['y_frac'] - c['y_frac']) < P.crowd_in for o in strong):
            continue
        out.append(c)
    return out


def decide(cands, wi, hi, P=FoldParams(), partner=None):
    """Fold decision from stage-1 candidates (`line_candidates`).  `partner`: stage-1 candidates of the other side of the sheet (or None).

    tier "A" (single page): score >= 1 (all criteria met) and the line sits at a customary fold position (prior_pos +- prior_tol);
    tier "B" (duplex corroboration): score >= pair_score_min and the partner side has a line of score >= pair_partner_min at the same
    relative row (+- pair_tol): a crease runs through the whole sheet, so it shows on both sides at the same height.
    tier "F" (tri-fold family): a line of score >= family_score_min exactly family_spacing (1/3) of the page height from a tier-A fold.
    Lines with another strong line close by (rulings, tables, grids) are dropped first."""
    if wi < P.min_page_in[0] or hi < P.min_page_in[1]:
        return []
    cands = _uncrowded(cands, P)
    pq = [q for q in _uncrowded(partner or [], P) if q['score'] >= P.pair_partner_min]
    out = []
    for c in cands:
        sc = c['score']
        tier = None
        if sc >= 1.0 and _near_prior(c['y_frac'], P):
            tier = "A"
        elif sc >= P.pair_score_min and any(abs(q['y_frac'] - c['y_frac']) <= P.pair_tol for q in pq):
            tier = "B"
        if tier:
            st = c['stats']
            out.append(Fold(c['y_frac'], c['angle'], c['amp'], st['pres'], (st['rel0'], st['rel1']), int(st['n_obs']), dict(score=sc, tier=tier, cont=st.get('cont'))))
    # tier F: the other creases of a tri-fold: a tier-A fold plus a weaker line exactly family_spacing of the page height away
    for a in [f for f in out if f.info['tier'] == 'A']:
        for c in cands:
            if c['score'] >= P.family_score_min and all(abs(c['y_frac'] - f.y_frac) > P.sep_in / hi for f in out) \
                    and abs(abs(c['y_frac'] - a.y_frac) - P.family_spacing) <= P.family_tol:
                st = c['stats']
                out.append(Fold(c['y_frac'], c['angle'], c['amp'], st['pres'], (st['rel0'], st['rel1']), int(st['n_obs']),
                                dict(score=c['score'], tier="F", cont=st.get('cont'))))
    return out


def _fold_geometry(rgb, dpi, f, P):
    """Native-resolution rows around one fold: (r0, r1, dens, coloured-ink mask, valid, trajectory in crop coordinates)."""
    h, w = rgb.shape[:2]
    y0 = f.y_frac * h
    r0, r1 = _fold_rows(h, w, y0, f.angle_deg, dpi, P)
    dens, cink, _ = density_maps(rgb[r0:r1], dpi, P, native=True)
    rs = max(1, int(round(P.strong_dil_in * dpi)))
    inkish = ((dens > P.strong) | cink).astype(np.uint8)
    valid = 1.0 - cv2.dilate(inkish, np.ones((2 * rs + 1, 2 * rs + 1), np.uint8)).astype(np.float32)
    traj = track(dens, valid, dpi, y0 - r0, f.angle_deg, P)
    return r0, r1, dens, cink, valid, traj


def rule_fraction(dens, traj, dpi, P=FoldParams()):
    """Native check: fraction of the columns where the line is a dark, isolated rule (density >= rule_dark within 2 rows of the trajectory and no
    ink in the rows 0.03..0.05 in above and below).  A crease is never that dark (<= ~0.45 at 300 ppi) and text crossing it has ink beside it."""
    h, w = dens.shape
    xs = np.arange(w)
    yi = np.round(traj).astype(np.int64)
    def at(off):
        return dens[np.clip(yi + off, 0, h - 1), xs]
    core = np.maximum.reduce([at(o) for o in (-2, -1, 0, 1, 2)])
    r_in, r_out = int(round(0.03 * dpi)), int(round(0.05 * dpi))
    side = np.maximum.reduce([at(o) for o in range(r_in, r_out + 1)] + [at(-o) for o in range(r_in, r_out + 1)])
    return float(((core >= P.rule_dark) & (side < P.rule_iso_d)).mean())


def detect_folds(img, dpi, P=FoldParams(), partner=None, native_check=True):
    """img: uint8 gray or RGB at its native resolution; partner: `line_candidates(...)[0]` of the sheet's other side (optional).
    native_check: drop lines that are dark isolated rules at native resolution (rule_fraction)."""
    cands, (wi, hi) = line_candidates(img, dpi, P)
    folds = decide(cands, wi, hi, P, partner)
    if native_check and img.ndim == 3:
        keep = []
        for f in folds:
            _, _, dens, _, _, traj = _fold_geometry(img, dpi, f, P)
            f.info['rule_frac'] = rf = rule_fraction(dens, traj, dpi, P)
            if rf <= P.rule_frac_max:
                keep.append(f)
        folds = keep
    return folds


# ------------------------------------------------------------------------------------------------ trajectory + mask (native resolution)
def _strip_coords(w, y0, t, jj, xs):
    """map_x, map_y for cv2.remap: row j of the strip at column x samples the source at (x, y0 + t*(x - w/2) + j)."""
    X = np.broadcast_to(xs[None, :], (len(jj), len(xs))).astype(np.float32)
    Y = (y0 + t * (xs[None, :] - w / 2.0) + jj[:, None]).astype(np.float32)
    return X, Y


def track(dens, valid, dpi, y0, angle_deg, P=FoldParams()):
    """Trajectory of the crease: row y(x) (float array over the columns of `dens`, native resolution) found by dynamic programming over column
    blocks inside +-traj_half_in of the Radon line (row y0 at the page centre, tilt angle_deg).  `valid` (float 0/1) masks ink: ink and its skirt
    must not attract the path (the crease can run along a text line)."""
    h, w = dens.shape
    t = np.tan(np.radians(angle_deg))
    J = max(3, int(round(P.traj_half_in * dpi)))
    jj = np.arange(-J, J + 1, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    X, Y = _strip_coords(w, y0, t, jj, xs)
    S = cv2.remap(np.minimum(dens, P.cap) * valid, X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    S = cv2.GaussianBlur(S, (0, 0), sigmaX=0.1, sigmaY=max(0.6, 0.0033 * dpi))
    bs = max(2, int(round(P.traj_block_in * dpi)))
    nb = w // bs
    B = S[:, :nb * bs].reshape(len(jj), nb, bs).mean(2)                 # (states, blocks)
    B = B - B.mean(0, keepdims=True)
    st = P.traj_step
    moves = [0] + [s_ * m for m in range(1, st + 1) for s_ in (1, -1)]
    pen = np.array([0.0] + [0.002 * m for m in range(1, st + 1) for _ in (0, 1)], np.float32)
    NEG = np.float32(-1e9)
    score = B[:, 0].copy()
    back = np.zeros((len(jj), nb), np.int16)
    cand = np.empty((len(moves), len(jj)), np.float32)
    for b in range(1, nb):
        for mi, m in enumerate(moves):
            cand[mi] = NEG
            if m == 0:
                cand[mi] = score
            elif m > 0:
                cand[mi, m:] = score[:-m] - pen[mi]
            else:
                cand[mi, :m] = score[-m:] - pen[mi]
        mi = cand.argmax(0)
        score = cand[mi, np.arange(len(jj))] + B[:, b]
        back[:, b] = np.array(moves)[mi]
    j = int(score.argmax())
    path = np.empty(nb, np.float32)
    for b in range(nb - 1, -1, -1):
        path[b] = jj[j]
        j = j - back[j, b]
    xc = (np.arange(nb) + 0.5) * bs
    jx = np.interp(xs, xc, path)
    jx = cv2.GaussianBlur(jx.reshape(1, -1), (0, 0), sigmaX=bs * 1.0).ravel()
    return y0 + t * (xs - w / 2.0) + jx


def _fold_rows(h, w, y0, angle_deg, dpi, P):
    """Row range of the page that the processing of one fold needs."""
    pad = abs(np.tan(np.radians(angle_deg))) * w / 2.0 + (P.traj_half_in + P.band_in + P.stroke_reach_in + 0.12) * dpi
    return max(0, int(y0 - pad)), min(h, int(y0 + pad) + 1)


def fold_mask(rgb, dpi, folds, P=FoldParams(), protect=None):
    """Crease pixel mask (uint8 H x W) at the resolution of `rgb`.  Band of +-band_in around the tracked trajectory, minus ink: pixels with
    density >= tone_dark (and their skirt, halo_in), coloured-ink pixels, and pixels on a vertical run of density >= stroke_t that reaches
    stroke_reach_in beyond the band on either side (a stroke crossing the crease), spread sideways by protect_dil_in.
    `protect`: optional bool/uint8 plane (H x W) of pixels that must never be removed (e.g. confident OCR word boxes).
    Only the rows around each fold are touched (the rest of the page is never read)."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bw = max(1, int(round(P.band_in * dpi)))
    reach = max(2, int(round(P.stroke_reach_in * dpi)))
    rh = max(1, int(round(P.halo_in * dpi)))
    for f in folds:
        r0, r1, dens, cink, valid, traj = _fold_geometry(rgb, dpi, f, P)
        ys = np.arange(dens.shape[0])[:, None]
        dist = ys - traj[None, :]
        band = np.abs(dist) <= bw
        ext = _runs_beyond((dens >= P.stroke_t).astype(np.uint8), dist, bw, reach)
        if P.protect_dil_in > 0:
            r = max(1, int(round(P.protect_dil_in * dpi)))
            ext = cv2.dilate(ext.astype(np.uint8), np.ones((1, 2 * r + 1), np.uint8)) > 0
        dark = ((dens >= P.tone_dark) | (cink if P.protect_chroma else False)).astype(np.uint8)
        keep = ext | (cv2.dilate(dark, np.ones((2 * rh + 1, 2 * rh + 1), np.uint8)) > 0)      # ink and its anti-aliased skirt
        if P.orient_h_min > 0:
            keep |= _not_horizontal(dens, dpi, P)
        if protect is not None:
            keep |= protect[r0:r1] > 0
        mask[r0:r1] |= (band & ~keep).astype(np.uint8)
    return mask


def _not_horizontal(dens, dpi, P):
    """True where the local structure of the density is not horizontal (structure tensor): strokes crossing the crease at any angle."""
    g = cv2.GaussianBlur(dens, (0, 0), max(0.6, 0.0033 * dpi))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    st = max(1.0, P.orient_sigma_in * dpi)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), st)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), st)
    e = jxx + jyy
    h = jyy / np.maximum(e, 1e-12)
    return (h < P.orient_h_min) & (e >= P.orient_e0)


def _runs_beyond(dense, dist, bw, reach):
    """Pixels of `dense` (uint8 strip) whose vertical run of ones contains a pixel at least `reach` rows beyond the band (|dist| >= bw + reach),
    above or below: a stroke that crosses the crease."""
    D = np.ascontiguousarray(dense.T)                     # (w, h): runs along the last axis
    st = D.copy()
    st[:, 1:] &= (1 - D[:, :-1])
    lab = np.cumsum(st.ravel().astype(np.int64)).reshape(D.shape) * D
    far = (np.abs(dist.T) >= (bw + reach)) & (D > 0)
    flag = np.zeros(int(lab.max()) + 1, bool)
    flag[lab[far]] = True
    flag[0] = False
    return flag[lab].T


def remove_folds(rgb, dpi, folds, P=FoldParams(), mask=None):
    """Return a copy of `rgb` with the crease pixels replaced by local paper: per column, the per-channel max over the unmasked pixels within
    bw + 6 rows (the paper just beside the band; never darker than either side), smoothed sideways by a 5 px max filter."""
    out = rgb.copy()
    if not folds:
        return out
    if mask is None:
        mask = fold_mask(rgb, dpi, folds, P)
    rows = np.flatnonzero(mask.any(1))
    if rows.size == 0:
        return out
    bw = max(1, int(round(P.band_in * dpi)))
    top, bot = max(0, rows.min() - bw - 6), min(rgb.shape[0], rows.max() + bw + 7)
    m = mask[top:bot] > 0
    sub = rgb[top:bot]
    big = cv2.dilate(m.astype(np.uint8), np.ones((2 * 3 + 1, 1), np.uint8)) > 0
    src = np.where(~big[..., None], sub, 0).astype(np.uint8)
    k = 2 * (bw + 6) + 1
    fill = cv2.dilate(cv2.dilate(src, np.ones((k, 1), np.uint8)), np.ones((1, 5), np.uint8))
    sub2 = sub.copy()
    sub2[m] = fill[m]
    out[top:bot] = sub2
    return out
