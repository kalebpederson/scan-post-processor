"""Pair-detection experiments on 50 ppi ink maps."""
import numpy as np, cv2, json, os
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
D50 = HERE + '/work/d50/'
WS = 50.0

def load(i): return np.load(f'{D50}{i:04d}.npy').astype(np.float32)

def dir_score(A, Bm, rng_in=1.0, faint_cap=0.25, real_thr=0.25, bcap=0.6, edge=0.1):
    """A receives bleed from Bm (already mirrored). returns z, peak, dx, dy, border (A(x,y) ~ Bm(x+dx,y+dy))."""
    h, w = A.shape
    R = int(rng_in * WS)
    real = cv2.dilate((A > real_thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    F = np.where(real, 0, np.clip(A, 0, faint_cap)).astype(np.float32)
    F[:, :int(edge * w)] = 0; F[:, int((1 - edge) * w):] = 0
    Bc = np.clip(Bm, 0, bcap).astype(np.float32)
    Bp = Bc
    T = np.ascontiguousarray(F[R:h - R, R:w - R])
    if T.std() < 1e-4: return dict(z=0.0, peak=0.0, dx=0, dy=0, border=True)
    res = cv2.matchTemplate(Bp, T, cv2.TM_CCOEFF_NORMED)
    i = np.unravel_index(res.argmax(), res.shape)
    z = (res.max() - res.mean()) / (res.std() + 1e-9)
    border = bool(i[0] in (0, res.shape[0] - 1) or i[1] in (0, res.shape[1] - 1))
    r2 = res.copy(); y0, x0 = i
    r2[max(0, y0 - 4):y0 + 5, max(0, x0 - 4):x0 + 5] = -1
    second = float(r2.max())
    return dict(z=float(z), peak=float(res.max()), dx=int(i[1] - R), dy=int(i[0] - R), border=border, second=second)

def common(A, B):
    h = min(A.shape[0], B.shape[0]); w = min(A.shape[1], B.shape[1])
    return A[:h, :w], B[:h, :w]

def pair_scores(A, B, **kw):
    A, B = common(A, B)
    a = dir_score(A, B[:, ::-1], **kw)
    b = dir_score(B, A[:, ::-1], **kw)
    return a, b

def shift_to(Bm, dx, dy, shape):
    """aligned[x,y] = Bm[x+dx, y+dy], zero outside"""
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(Bm, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)

def extra_feats(A, Bm, dx, dy, thr=0.3):
    h, w = A.shape
    Bs = shift_to(Bm, dx, dy, A.shape)
    valid = np.zeros((h, w), bool); valid[int(.04*h):int(.96*h), int(.1*w):int(.9*w)] = True
    SA = (A > thr) & valid; SB = (Bs > thr) & valid
    nA, nB, N = SA.sum(), SB.sum(), valid.sum()
    k = np.ones((3, 3), np.uint8)
    SAd = cv2.dilate(SA.astype(np.uint8), k) > 0; SBd = cv2.dilate(SB.astype(np.uint8), k) > 0
    inter = (SAd & SBd).sum()
    exp = SAd.sum() * SBd.sum() / max(N, 1)
    coin = float(inter / (exp + 1e-6)) if nA > 50 and nB > 50 else 0.0
    fracB_in_A = float((SB & SAd).sum() / max(nB, 1))
    # bleed regression: F_A vs footprint of B's strong ink (blurred)
    real = cv2.dilate((A > 0.25).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    foot = cv2.GaussianBlur(SB.astype(np.float32), (0, 0), 1.0)
    on = valid & ~real & (foot > 0.5); off = valid & ~real & (foot < 0.02) & (cv2.dilate(SB.astype(np.uint8), np.ones((9, 9), np.uint8)) == 0)
    on_m = float(A[on].mean()) if on.sum() > 30 else 0.0
    off_m = float(A[off].mean()) if off.sum() > 30 else 0.0
    return dict(coin=coin, nA=int(nA), nB=int(nB), on=on_m, off=off_m, non=int(on.sum()), contrast=on_m - off_m, fracB_in_A=fracB_in_A)
