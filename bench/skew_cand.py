"""Candidate skew-voter implementations tried on the R4 ground truth (experiments; see skew_variants.py)."""
import numpy as np, cv2
from scanpp import geometry as G

def ens(e):
    v = np.array([e[k] for k in G._VOTE_ORDER]); best = None
    for j in range(4):
        s = np.abs(v - v[j]) < G.VOTE_TOL
        if best is None or s.sum() > best.sum(): best = s
    w = G._VOTE_W * best
    ang = float((v * w).sum() / w.sum()); conf = (int(best.sum()) - 1) / 3.0
    if abs(ang) > G.MAX_ANGLE: conf = 0.0
    return ang, conf

def crop(gray, dpi):
    g, d = G.prep_gray(gray, dpi); h, w = g.shape
    mx, my = int(round(0.5 * d)), int(round(0.25 * d))
    return np.ascontiguousarray(g[my:h - my, mx:w - mx]), d

def make(proj=G._skew_projection, lsd=G._skew_lsd, cc=G._skew_cc, grad=G._skew_grad):
    def f(gray, dpi):
        g, d = crop(gray, dpi); rng = G.MAX_ANGLE + 1.0
        e = dict(proj=proj(g, d, rng), lsd=lsd(g, d, rng), cc=cc(g, d, rng), grad=grad(g, d, rng))
        return dict(e=e, ens=ens(e))
    return f

def _var(im, a):
    s = cv2.reduce(G._rot_gray(im, a), 1, cv2.REDUCE_SUM, dtype=cv2.CV_64F).ravel()
    return float(s.var())

def proj_fast(g, dpi, rng):      # same search, cheaper variance
    ink = G.ink_map(g, dpi); ink = np.where(ink > 0.12, ink, 0).astype(np.float32)
    sm = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    angs = np.arange(-rng, rng + 1e-9, 0.5); a0 = angs[np.argmax([_var(sm, a) for a in angs])]
    angs2 = np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.1); a1 = angs2[np.argmax([_var(ink, a) for a in angs2])]
    angs3 = np.arange(a1 - 0.1, a1 + 0.1 + 1e-9, 0.02)
    return float(angs3[np.argmax([_var(ink, a) for a in angs3])])

def proj_half_mid(g, dpi, rng):  # mid stage at half-res
    ink = G.ink_map(g, dpi); ink = np.where(ink > 0.12, ink, 0).astype(np.float32)
    sm = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    angs = np.arange(-rng, rng + 1e-9, 0.5); a0 = angs[np.argmax([_var(sm, a) for a in angs])]
    angs2 = np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.1); a1 = angs2[np.argmax([_var(sm, a) for a in angs2])]
    angs3 = np.arange(a1 - 0.1, a1 + 0.1 + 1e-9, 0.02)
    return float(angs3[np.argmax([_var(ink, a) for a in angs3])])

def proj_parab(g, dpi, rng):     # fine stage: 5 points at 0.05 + parabola
    ink = G.ink_map(g, dpi); ink = np.where(ink > 0.12, ink, 0).astype(np.float32)
    sm = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    angs = np.arange(-rng, rng + 1e-9, 0.5); a0 = angs[np.argmax([_var(sm, a) for a in angs])]
    angs2 = np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, 0.1); v2 = np.array([_var(ink, a) for a in angs2]); j = int(v2.argmax())
    a1 = angs2[j]
    angs3 = np.arange(a1 - 0.1, a1 + 0.1 + 1e-9, 0.05); v3 = np.array([_var(ink, a) for a in angs3]); j = int(v3.argmax())
    if 0 < j < len(v3) - 1:
        y0, y1, y2 = v3[j - 1], v3[j], v3[j + 1]; den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(angs3[j] + off * 0.05)
    return float(angs3[j])

def _parab(angs, v):
    j = int(np.argmax(v))
    if 0 < j < len(v) - 1:
        y0, y1, y2 = v[j - 1], v[j], v[j + 1]; den = y0 - 2 * y1 + y2
        if den < 0:
            return float(angs[j] + 0.5 * (y0 - y2) / den * (angs[1] - angs[0]))
    return float(angs[j])

def proj_gen(mid_step, mid_half, mid_span, fine_step, fine_span, u8=False, mid_parab=True):
    def f(g, dpi, rng):
        ink = G.ink_map(g, dpi); ink = np.where(ink > 0.12, ink, 0).astype(np.float32)
        if u8: ink = np.clip(ink * 255 + .5, 0, 255).astype(np.uint8)
        sm = cv2.resize(ink, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        angs = np.arange(-rng, rng + 1e-9, 0.5); a0 = angs[np.argmax([_var(sm, a) for a in angs])]
        src = sm if mid_half else ink
        if mid_step:
            angs2 = np.arange(a0 - mid_span, a0 + mid_span + 1e-9, mid_step); v2 = np.array([_var(src, a) for a in angs2])
            a1 = _parab(angs2, v2) if mid_parab else angs2[int(v2.argmax())]
        else: a1 = a0
        angs3 = np.arange(a1 - fine_span, a1 + fine_span + 1e-9, fine_step)
        return _parab(angs3, np.array([_var(ink, a) for a in angs3]))
    return f

def lsd_none(g, dpi, rng):
    return G._skew_lsd(g, dpi, rng)

def proj_sparse(coarse_step=0.5, mid_step=0.2, fine_step=0.05, nearest=False):
    def f(g, dpi, rng):
        ink = G.ink_map(g, dpi)
        ys, xs = np.nonzero(ink > 0.12)
        h, w = ink.shape
        v = ink[ys, xs].astype(np.float32)
        yc = ys.astype(np.float32) - (h / 2); xc = xs.astype(np.float32) - (w / 2)
        def var(a):
            r = np.radians(a); yy = yc * np.float32(np.cos(r)) + xc * np.float32(np.sin(r)) + np.float32(h / 2)
            i = yy.astype(np.int32); fr = yy - i
            ok = (i >= 0) & (i < h - 1)
            i = i[ok]; fr = fr[ok]; vv = v[ok]
            hist = np.bincount(i, vv * (1 - fr), minlength=h + 1) + np.bincount(i + 1, vv * fr, minlength=h + 1)
            return float(hist[:h].var())
        angs = np.arange(-rng, rng + 1e-9, coarse_step); a0 = angs[np.argmax([var(a) for a in angs])]
        angs2 = np.arange(a0 - 0.6, a0 + 0.6 + 1e-9, mid_step); a1 = _parab(angs2, np.array([var(a) for a in angs2]))
        angs3 = np.arange(a1 - 0.1, a1 + 0.1 + 1e-9, fine_step)
        return _parab(angs3, np.array([var(a) for a in angs3]))
    return f

def lsd_gen(refine, scale=1.0, minlen=0.3):
    def f(g, dpi, rng):
        gg, dd = g, dpi
        if scale != 1.0:
            gg = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA); dd = dpi * scale
        L = cv2.createLineSegmentDetector(refine).detect(gg)[0]
        if L is None: return 0.0
        L = L.reshape(-1, 4)
        dx, dy = L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]
        ln = np.hypot(dx, dy); k = ln >= dd * minlen
        dx, dy, ln = dx[k], dy[k], ln[k]
        ang = (np.degrees(np.arctan2(-dy, dx)) + 90) % 180 - 90
        dev = np.where(np.abs(ang) <= 45, ang, ang - np.sign(ang) * 90)
        m = np.abs(dev) <= rng
        return G._peak_hist(dev[m], ln[m] ** 2, -rng, rng, 0.1, 0.5)[0]
    return f

B = proj_gen(0.2, False, 0.6, 0.05, 0.1)
Bu = proj_gen(0.2, False, 0.6, 0.05, 0.1, u8=True)
VARIANTS = {
    "cur": make(),
    "lsd_.6": make(lsd=lsd_gen(cv2.LSD_REFINE_STD, 0.6)),
    "lsd_.5": make(lsd=lsd_gen(cv2.LSD_REFINE_STD, 0.5)),
    "lsd_.5_none": make(lsd=lsd_gen(cv2.LSD_REFINE_NONE, 0.5)),
}
