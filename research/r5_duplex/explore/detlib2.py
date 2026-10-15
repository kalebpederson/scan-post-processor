import numpy as np, cv2, os
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
def load_r(i):
    return np.load(f'{HERE}/work/r50/{"b%03d" % i if isinstance(i, str) is False and False else i:04d}.npy') if isinstance(i, int) else np.load(f'{HERE}/work/r50/{i}.npy')
def maps(r8):
    r = r8.astype(np.float32) / 255
    d = r.max(-1); ch = d - r.min(-1)
    return d, ch
def sink_layer(d, real_thr=0.25, cap=0.25, edge=0.1):
    h, w = d.shape
    real = cv2.dilate((d > real_thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    F = np.where(real, 0, np.clip(d, 0, cap)).astype(np.float32)
    F[:, :int(edge * w)] = 0; F[:, int((1 - edge) * w):] = 0
    return F
def src_layer(m, cap=0.6, mirror=True):
    S = np.clip(m, 0, cap).astype(np.float32)
    return np.ascontiguousarray(S[:, ::-1]) if mirror else S
def tm(F, S, R=50):
    h, w = F.shape
    T = np.ascontiguousarray(F[R:h - R, R:w - R])
    if T.std() < 1e-4 or S.std() < 1e-4: return dict(z=0.0, peak=0.0, second=0.0, dx=0, dy=0, border=True)
    res = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED)
    i = np.unravel_index(res.argmax(), res.shape)
    z = (res.max() - res.mean()) / (res.std() + 1e-9)
    r2 = res.copy(); y0, x0 = i
    r2[max(0, y0 - 4):y0 + 5, max(0, x0 - 4):x0 + 5] = -1
    return dict(z=float(z), peak=float(res.max()), second=float(r2.max()), dx=int(i[1] - R), dy=int(i[0] - R),
                border=bool(i[0] in (0, res.shape[0] - 1) or i[1] in (0, res.shape[1] - 1)))
def crop2(a, b):
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1]); return a[:h, :w], b[:h, :w]
def variants(rA, rB):
    """returns dict variant -> (a: A receives from B, b: B receives from A)"""
    dA, cA = maps(rA); dB, cB = maps(rB)
    dA, dB = crop2(dA, dB); cA, cB = crop2(cA, cB)
    out = {}
    FA, FB = sink_layer(dA), sink_layer(dB)
    out['d_d'] = (tm(FA, src_layer(dB)), tm(FB, src_layer(dA)))
    out['d_c'] = (tm(FA, src_layer(cB)), tm(FB, src_layer(cA)))
    out['d_cb'] = (tm(FA, src_layer(cv2.GaussianBlur(cB, (0, 0), 1.5) * 1.0)), tm(FB, src_layer(cv2.GaussianBlur(cA, (0, 0), 1.5))))
    return out

def variants2(rA, rB, names=('c_c', 'c_cs')):
    dA, cA = maps(rA); dB, cB = maps(rB)
    dA, dB = crop2(dA, dB); cA, cB = crop2(cA, cB)
    out = {}
    def sinkc(d, c, thr=0.25, cap=0.25):
        h, w = d.shape
        real = cv2.dilate((d > thr).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        F = np.where(real, 0, np.clip(c, 0, cap)).astype(np.float32); F[:, :int(.1 * w)] = 0; F[:, int(.9 * w):] = 0
        return F
    FA, FB = sinkc(dA, cA), sinkc(dB, cB)
    out['c_c'] = (tm(FA, src_layer(cB)), tm(FB, src_layer(cA)))
    # strong chroma sources only (marker / pen / highlighter bodies): chroma > 0.3
    sA = np.where(cA > 0.3, cA, 0).astype(np.float32); sB = np.where(cB > 0.3, cB, 0).astype(np.float32)
    out['c_cs'] = (tm(FA, src_layer(sB)), tm(FB, src_layer(sA)))
    return out
