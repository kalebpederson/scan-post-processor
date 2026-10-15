"""Scan cleaning: flat-field background division + hysteresis/sharpness soft ink gate + hue-preserving faint-ink boost.

    clean_page(rgb, dpi, paper="flatten", **params) -> uint8 HxWx3

Pipeline (all lengths in inches, so results are dpi-independent; native resolution in, native resolution out)
 1. Background (paper colour + illumination) per channel on a ~50 dpi copy: small grey opening (kills bright JPEG-overshoot specks
    next to ink) -> local max (grey dilation, window `bg_win_in`) -> guard (`blob_guard`): never let the background fall below
    `blob_guard` x the median background of the surrounding `guard_win_in` window, so big dark/colour blobs are not mistaken for paper
    -> Gaussian blur.  Works for coloured paper and uneven lighting.
 2. r_c = max(0, 1 - rgb_c/bg_c)  ("ink density" per channel; paper = 0).   d = max_c r_c  (any ink, incl. pale highlighter),
    chroma = max r - min r.
 3. soft gate alpha(x) in [0,1] = smoothstep((u - t_lo)/(t_hi - t_lo)) on u = blur(d) + sharp_w * |grad blur(d)|
    * page-adaptive noise floor (`noise_k`): t_lo is raised to median+noise_k*MAD of the paper's d (textured/pink/gray paper).
    * hysteresis (`gate="hyst"`): connected components of {d > t_lo}; a component that touches a strong pixel (d > t_seed) or is large
      (> comp_min_in2 square inches) is "supported" and uses the low threshold; isolated weak components (bleed-through ghosts,
      speckle) need the higher threshold t_hi_unsup.  The ramp is soft everywhere (no hard decision on pixels).
 4. ink tone: dk_c = clip(r_c/(1-bp) * k, 0, 1), k = 1 + (faint_gain-1) * fade(d) * fade(chroma)  -> darkens faint grey graphite /
    light print with a *common* factor on all channels (hue preserved); coloured ink (highlighter, pens) is not boosted.
 5. out = 1 - alpha*dk  (paper -> pure white).  paper="keep": out *= global paper colour (uniform tint; shading, bleed, noise removed).
Other gates: gate="none" (plain soft threshold = 'M2'), "sharp" (sharpness bonus only), "m1" (flat-field + per-channel levels).
"""
import numpy as np, cv2

DEFAULTS = dict(bg_win_in=0.35, guard_win_in=1.5, blob_guard=0.8, bg_open_in=0.06, bg_work_dpi=50.0,
                t_lo=0.02, t_hi=0.14, bp=0.16, gamma=1.0,
                gate="hyst", t_seed=0.25, t_hi_unsup=0.12, comp_min_in2=0.5, denoise_in=0.006,
                sharp_w=0.5, sharp_sigma_in=0.0033, sharp_dil_in=0.0, noise_k=3.0, wp=0.9,
                faint_gain=3.0, faint_lo=0.05, faint_hi=0.25, faint_chroma_lo=0.08, faint_chroma_hi=0.22,
                line_support=False, t_line=0.03, line_len_in=0.4, line_gap_in=0.06)

def estimate_background(rgb, dpi, bg_win_in, guard_win_in, blob_guard, bg_work_dpi, bg_open_in=0.0):
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
        med = cv2.medianBlur(cv2.resize(bg, (max(8, sw // 2), max(8, sh // 2)), interpolation=cv2.INTER_AREA), max(3, (kg // 2) | 1))
        med = cv2.resize(med, (sw, sh), interpolation=cv2.INTER_LINEAR)
        bg = np.maximum(bg, (med.astype(np.float32) * blob_guard).astype(np.uint8))
    return cv2.GaussianBlur(bg.astype(np.float32), (0, 0), k / 2.5)

def _smoothstep(x):            # in place on float32 array
    np.clip(x, 0, 1, out=x)
    t = 3.0 - 2.0 * x
    x *= x; x *= t
    return x

def _line_mask(dm, dpi, thr, length_in, gap_in, work_dpi=100.0):
    h, w = dm.shape; s = min(1.0, work_dpi / dpi)
    sm = cv2.resize(((dm > thr).astype(np.uint8) * 255), (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA) > 60
    sm = sm.astype(np.uint8); g = max(1, int(round(gap_in * work_dpi))); L = max(3, int(round(length_in * work_dpi))); out = np.zeros_like(sm)
    for kc, ko in ((cv2.getStructuringElement(cv2.MORPH_RECT, (g, 1)), cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))),
                   (cv2.getStructuringElement(cv2.MORPH_RECT, (1, g)), cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))):
        out |= cv2.morphologyEx(cv2.morphologyEx(sm, cv2.MORPH_CLOSE, kc), cv2.MORPH_OPEN, ko)
    out = cv2.dilate(out, np.ones((3, 3), np.uint8))
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST) > 0

def clean_page(rgb, dpi, paper="flatten", return_debug=False, **params):
    P = dict(DEFAULTS); P.update(params)
    if paper not in ("flatten", "keep"): raise ValueError("paper must be 'flatten' or 'keep'")
    h, w = rgb.shape[:2]
    bgs = np.maximum(estimate_background(rgb, dpi, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], P["bg_work_dpi"], P["bg_open_in"]), 16.0)
    lumbg = bgs.mean(-1); paper_col = np.median(bgs[lumbg >= np.percentile(lumbg, 50)], axis=0)   # global paper colour (brighter half of page)
    # planar float32 ink density r_c = max(0, 1 - rgb_c / bg_c)
    r = []
    for c in range(3):
        bgc = cv2.resize(np.ascontiguousarray(bgs[..., c]), (w, h), interpolation=cv2.INTER_LINEAR)
        rc = cv2.subtract(bgc, rgb[..., c].astype(np.float32)); cv2.divide(rc, bgc, rc); np.maximum(rc, 0, out=rc); r.append(rc)
    d = cv2.max(cv2.max(r[0], r[1]), r[2])
    rmin = cv2.min(cv2.min(r[0], r[1]), r[2])
    dm = cv2.GaussianBlur(d, (0, 0), max(0.5, P["denoise_in"] * dpi)) if P["denoise_in"] > 0 else d
    t_lo, t_hi, t_unsup = P["t_lo"], P["t_hi"], P["t_hi_unsup"]
    if P["gate"] == "m1":
        out = [np.clip(1.0 - rc - P["bp"], 0, 1) / (P["wp"] - P["bp"]) for rc in r]
        out = [np.clip(o, 0, 1) ** P["gamma"] for o in out]
        alpha = None
    else:
        if P["noise_k"] > 0:     # page-adaptive noise floor
            sub = dm[::4, ::4]; pp = sub[sub < 0.12]
            if pp.size > 1000:
                f = float(np.median(pp)); sig = 1.4826 * float(np.median(np.abs(pp - f))); floor = f + P["noise_k"] * sig
                if floor > t_lo:
                    t_hi += floor - t_lo; t_unsup = max(t_unsup, floor + (P["t_hi_unsup"] - P["t_lo"])); t_lo = floor
        u = dm
        if P["gate"] in ("hyst", "sharp") and P["sharp_w"] > 0:
            g1 = cv2.GaussianBlur(d, (0, 0), max(0.6, P["sharp_sigma_in"] * dpi))
            gx = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
            E = cv2.magnitude(gx, gy)
            if P["sharp_dil_in"] > 0:
                kk = max(1, int(round(P["sharp_dil_in"] * dpi)) | 1); E = cv2.dilate(E, np.ones((kk, kk), np.uint8))
            u = dm + E * (P["sharp_w"] / 8.0)
        if P["gate"] == "hyst":
            nl, lab = cv2.connectedComponents((dm > t_lo).astype(np.uint8), connectivity=8)
            has = np.zeros(nl, bool); has[np.unique(lab[dm > P["t_seed"]])] = True; has[0] = False
            if P["comp_min_in2"] > 0:
                has |= (np.bincount(lab.ravel(), minlength=nl) > P["comp_min_in2"] * dpi * dpi)
            sup = has[lab]
            if P["line_support"]: sup |= _line_mask(dm, dpi, P["t_line"], P["line_len_in"], P["line_gap_in"])
            x = u - t_lo; x -= np.where(sup, 0.0, t_unsup - t_lo).astype(np.float32)
        else:
            x = u - t_lo
        x *= 1.0 / max(t_hi - t_lo, 1e-3)
        alpha = _smoothstep(x)
        k = None
        if P["faint_gain"] != 1.0:   # hue-preserving boost of faint grey ink
            fd = _smoothstep((dm - P["faint_lo"]) * (1.0 / (P["faint_hi"] - P["faint_lo"])))
            fc = _smoothstep((d - rmin - P["faint_chroma_lo"]) * (1.0 / (P["faint_chroma_hi"] - P["faint_chroma_lo"])))
            k = 1.0 + (P["faint_gain"] - 1.0) * (1.0 - fd) * (1.0 - fc)
        inv = 1.0 / (1.0 - P["bp"])
        out = []
        for rc in r:
            dk = rc * inv
            if k is not None: dk *= k
            np.minimum(dk, 1.0, out=dk)
            if P["gamma"] != 1.0: dk = 1.0 - (1.0 - dk) ** P["gamma"]
            dk *= alpha
            out.append(1.0 - dk)
    res = np.empty((h, w, 3), np.uint8)
    for c in range(3):
        o = out[c] * (255.0 * (paper_col[c] / 255.0 if paper == "keep" else 1.0)); o += 0.5
        res[..., c] = np.clip(o, 0, 255).astype(np.uint8)
    if return_debug: return res, dict(d=d, alpha=alpha, paper_col=paper_col)
    return res
