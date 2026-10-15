import numpy as np, cv2
from skimage.metrics import structural_similarity as ssim_fn
from skimage.color import rgb2lab, deltaE_ciede2000

WHITE_T = 245
def fit(img, shape):
    """crop/pad (white) render to reference shape (renderers may differ by a pixel)."""
    h, w = shape[:2]
    out = np.full((h, w, 3), 255, np.uint8); hh, ww = min(h, img.shape[0]), min(w, img.shape[1]); out[:hh, :ww] = img[:hh, :ww]; return out

def fidelity(ref, ren, chroma_thr=25.0):
    ren = fit(ren, ref.shape)
    gref = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY); gren = cv2.cvtColor(ren, cv2.COLOR_RGB2GRAY)
    s = float(ssim_fn(gref, gren, data_range=255))
    mn_ref = ref.min(-1); mn_ren = ren.min(-1)
    solid = mn_ref < 128; faint = (mn_ref >= 128) & (mn_ref < 200)
    white_ren = mn_ren >= WHITE_T
    loss_solid = float((solid & white_ren).sum() / max(1, solid.sum()))
    loss_faint = float((faint & white_ren).sum() / max(1, faint.sum()))
    refwhite = (ref == 255).all(-1)
    speck = float((refwhite & (mn_ren < WHITE_T)).sum() / max(1, refwhite.sum()))
    # colour accuracy over chroma>thr ink pixels (ref chroma by Lab C*)
    sub = ref.reshape(-1, 3)
    mx = ref.max(-1).astype(np.int16) - ref.min(-1).astype(np.int16)
    cand = (mx > 20) & (mn_ref < 200)
    ys, xs = np.nonzero(cand)
    if len(ys) > 150000:
        sel = np.random.default_rng(0).choice(len(ys), 150000, replace=False); ys, xs = ys[sel], xs[sel]
    if len(ys):
        lr = rgb2lab(ref[ys, xs][None] / 255.0)[0]; lo = rgb2lab(ren[ys, xs][None] / 255.0)[0]
        C = np.hypot(lr[:, 1], lr[:, 2]); m = C > chroma_thr
        de = float(deltaE_ciede2000(lr[m], lo[m]).mean()) if m.any() else float("nan")
        n_de = int(m.sum())
    else: de, n_de = float("nan"), 0
    return dict(ssim=s, dE=de, n_de=n_de, loss_solid=loss_solid, loss_faint=loss_faint, speckle=speck)

def renderer_diff(a, b):
    """mean abs diff and p99.9 of abs diff (max over channels) between two renders"""
    b = fit(b, a.shape)
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return float(d.mean()), float(np.percentile(d, 99.9))
