"""Exploratory stage-1 (Radon over near-horizontal lines of the strong-ink-masked ink density) + per-window span features.
Used for the corpus sweep; the shipped-style API is in fold.py."""
import numpy as np, cv2
from common import *
from scanpp.clean import estimate_background

def dmap(rgb, dpi):
    bg = estimate_background(rgb, dpi, 0.35, 1.5, 0.8, 50.0, 0.06)
    bg = cv2.resize(bg, (rgb.shape[1], rgb.shape[0]))
    g = rgb.astype(np.float32).mean(2)
    return np.clip(1 - g / np.maximum(bg.mean(2), 16), 0, 1)

def shear(a, t):
    H, W = a.shape
    M = np.array([[1, 0, 0], [t, 1, -t * W / 2]], np.float32)
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def prep(d, dpi, strong=0.5, cap=0.3, dil=0.02):
    S = (d > strong).astype(np.uint8); r = max(1, int(round(dil * dpi)))
    S = cv2.dilate(S, np.ones((2 * r + 1, 2 * r + 1), np.uint8))
    v = (1 - S).astype(np.float32)
    return np.minimum(d, cap) * v, v

def radon(dv, v, angs, xm=0.03):
    H, W = dv.shape; x0, x1 = int(W * xm), int(W * (1 - xm))
    R = np.empty((len(angs), H), np.float32); C = np.empty_like(R)
    for i, a in enumerate(angs):
        t = np.tan(np.radians(a))
        num = shear(dv, t)[:, x0:x1].sum(1); den = shear(v, t)[:, x0:x1].sum(1)
        R[i] = num / np.maximum(den, 1); C[i] = den / (x1 - x0)
    return R, C

def candidates(rgb, dpi, angs=np.arange(-3, 3.01, 0.25), n=8, cov_min=0.4, k=10):
    d = dmap(rgb, dpi); dv, v = prep(d, dpi)
    R, C = radon(dv, v, angs)
    base = cv2.blur(R.T, (1, 2 * k + 1)).T
    rid = (R - base) * (C > cov_min)
    best = rid.max(0); ai = rid.argmax(0); H = R.shape[1]
    sel = []
    for i in np.argsort(-best)[:200]:
        if best[i] <= 0: break
        if all(abs(i - j) > int(0.3 * dpi) for j in sel): sel.append(int(i))
        if len(sel) >= n: break
    out = []
    for i in sel:
        out.append(dict(y=i, yf=i / H, ang=float(angs[ai[i]]), amp=float(best[i]), cov=float(C[ai[i], i])))
    return out, (dv, v)

def window_profile(dv, v, y, ang, dpi, win_in=0.25, xm=0.0):
    """Per-window ridge strength in the frame sheared by `ang`: returns (z[nwin], ridge[nwin], noise[nwin], cov[nwin])."""
    H, W = dv.shape; t = np.tan(np.radians(ang))
    a = shear(dv, t); b = shear(v, t)
    ww = int(win_in * dpi); nw = W // ww
    lo, hi = max(0, y - int(0.3 * dpi)), min(H, y + int(0.3 * dpi))
    a = a[lo:hi, :nw * ww].reshape(hi - lo, nw, ww).sum(2); b = b[lo:hi, :nw * ww].reshape(hi - lo, nw, ww).sum(2)
    prof = a / np.maximum(b, 1)
    cov = b / ww
    r = int(0.05 * dpi) + 1  # +-3 rows search
    yy = y - lo
    base = np.concatenate([prof[max(0, yy - 9):max(0, yy - 4)], prof[yy + 5:yy + 10]], 0)
    bm = np.median(base, 0) if base.size else np.zeros(nw)
    rid = prof[max(0, yy - 2):yy + 3].max(0) - bm
    # noise: robust spread of the profile (3-row means) over the strip, per window
    sm = cv2.blur(prof, (1, 3))
    nz = 1.4826 * np.median(np.abs(sm - np.median(sm, 0)), 0) + 1e-4
    return rid / nz, rid, nz, cov[max(0, yy - 2):yy + 3].mean(0)  # (z, ridge, noise, valid-pixel coverage)
