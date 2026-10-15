"""Scan cleaning: flat-field background division + soft ink/background gating.

clean_page(rgb, dpi, paper="flatten", **params) -> uint8 HxWx3

Pipeline
 1. Background (paper + illumination) per channel, estimated on a ~50 dpi copy: local max (grey dilation, window `bg_win_in`)
    -> median-of-neighbourhood guard (`guard_win_in`, `blob_guard`) so big dark/colour blobs are not mistaken for paper -> Gaussian blur.
 2. n = rgb / bg  (paper ~ 1.0 for any paper colour / uneven light).
 3. d = 1 - min_c(n): "darkness of any ink" (also fires for pale yellow/pink highlighter because one channel drops).
 4. gate alpha(d) = smoothstep(t_lo .. t_hi) (soft background->white decision).  Optional support gating (`gate`):
      'hyst'  : weak pixels (d>t_lo) survive at the low threshold only if connected to strong (d>t_seed) pixels or their component
                is itself big/dark enough; else they need the higher threshold t_hi_unsup  (pencil/grid vs. bleed)
 5. ink tone: per channel t=((n-bp)/(1-bp))^gamma  (black point + gamma; hue preserved, black text -> black), out = 1 - alpha*(1 - t)
 6. paper="flatten": paper -> white.  paper="keep": multiply by global paper colour (uniform tint, no shading/bleed/noise).
All lengths are in inches (dpi-aware). Defaults = chosen operating point (see REPORT.md).
"""
import numpy as np, cv2

DEFAULTS = dict(bg_win_in=0.35, guard_win_in=1.5, blob_guard=0.8, bg_work_dpi=50.0,
                t_lo=0.10, t_hi=0.22, bp=0.12, gamma=1.0, sat_boost=1.0,
                gate="none", t_seed=0.30, t_hi_unsup=0.30, comp_min_in2=0.0, denoise_in=0.006, sharp_w=0.0, sharp_sigma_in=0.0033, sharp_dil_in=0.0, wp=0.9)

def estimate_background(rgb, dpi, bg_win_in, guard_win_in, blob_guard, bg_work_dpi):
    h, w = rgb.shape[:2]
    s = max(1.0, dpi / bg_work_dpi)
    sw, sh = max(8, int(round(w / s))), max(8, int(round(h / s)))
    small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(bg_win_in * bg_work_dpi)) | 1)
    bg = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))      # local max per channel (uint8)
    if blob_guard > 0:
        kg = max(5, int(round(guard_win_in * bg_work_dpi)) | 1)
        # median of the max-map over a big window ~ paper colour of this neighbourhood
        med = cv2.medianBlur(cv2.resize(bg, (max(8, sw // 2), max(8, sh // 2)), interpolation=cv2.INTER_AREA), max(3, (kg // 2) | 1))
        med = cv2.resize(med, (sw, sh), interpolation=cv2.INTER_LINEAR)
        bg = np.maximum(bg, (med.astype(np.float32) * blob_guard).astype(np.uint8))
    bgf = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), k / 2.5)
    return bgf, (sw, sh), small

def _smoothstep(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

def clean_page(rgb, dpi, paper="flatten", return_debug=False, **params):
    P = dict(DEFAULTS); P.update(params)
    h, w = rgb.shape[:2]
    bgs, (sw, sh), small = estimate_background(rgb, dpi, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], P["bg_work_dpi"])
    bgs = np.maximum(bgs, 16.0)
    # global paper colour = median of bg over the brighter half of the page (robust to shadows / non-paper)
    lumbg = bgs.mean(-1); sel = lumbg >= np.percentile(lumbg, 50)
    paper_col = np.median(bgs[sel], axis=0)
    bg = cv2.resize(bgs, (w, h), interpolation=cv2.INTER_LINEAR)
    n = np.minimum(rgb.astype(np.float32) / bg, 1.0)
    d = 1.0 - n.min(-1)
    # --- gate
    dm = d
    if P["denoise_in"] > 0:
        sg = max(0.5, P["denoise_in"] * dpi)
        dm = cv2.GaussianBlur(d, (0, 0), sg)
    t_lo, t_hi = P["t_lo"], P["t_hi"]
    sup_w = 0.0
    if P["gate"] == "m1":
        alpha = None
    elif P["gate"] == "hyst":
        weak = dm > t_lo
        strong = (dm > P["t_seed"]).astype(np.uint8)
        nl, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
        has = np.zeros(nl, bool); has[np.unique(lab[strong > 0])] = True; has[0] = False
        if P["comp_min_in2"] > 0:
            area = np.bincount(lab.ravel(), minlength=nl); has |= (area > P["comp_min_in2"] * dpi * dpi)
        delta = np.where(has[lab], 0.0, P["t_hi_unsup"] - t_lo).astype(np.float32)
        alpha = _smoothstep((dm - t_lo - delta) / max(t_hi - t_lo, 1e-3))
    elif P["gate"] == "sharp":
        sg1 = max(0.6, P["sharp_sigma_in"] * dpi)
        g1 = cv2.GaussianBlur(d, (0, 0), sg1)
        gx = cv2.Sobel(g1, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g1, cv2.CV_32F, 0, 1, ksize=3)
        E = np.sqrt(gx * gx + gy * gy) * (1.0 / 8.0)
        if P["sharp_dil_in"] > 0:
            kk = max(1, int(round(P["sharp_dil_in"] * dpi)) | 1); E = cv2.dilate(E, np.ones((kk, kk), np.uint8))
        u = dm + P["sharp_w"] * E
        alpha = _smoothstep((u - t_lo) / max(t_hi - t_lo, 1e-3))
    else:
        alpha = _smoothstep((dm - t_lo) / max(t_hi - t_lo, 1e-3))
    # --- tone
    bp = P["bp"]
    t = np.clip((n - bp) / (1 - bp), 0, 1)
    if P["gamma"] != 1.0: t = t ** P["gamma"]
    if P["sat_boost"] != 1.0:
        g = t.mean(-1, keepdims=True); t = np.clip(g + (t - g) * P["sat_boost"], 0, 1)
    if P["gate"] == "m1":
        t = np.clip((n - bp) / (P["wp"] - bp), 0, 1)
        if P["gamma"] != 1.0: t = t ** P["gamma"]
        out = t; alpha = np.ones(d.shape, np.float32)
    else:
        out = 1.0 - alpha[..., None] * (1.0 - t)
    if paper == "keep":
        out = out * (paper_col / 255.0)[None, None, :]
    res = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if return_debug: return res, dict(bg=bg, n=n, d=d, alpha=alpha, paper_col=paper_col)
    return res
