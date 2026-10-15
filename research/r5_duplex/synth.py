"""Synthetic overlap test.  Brother sheet 1 (A=front with marker, B=back with real ghost).  Real handwriting crops (blue/red/neutral from a 300 dpi
corpus page + the user's notes from B itself) are multiplied onto B (a) over the marker ghost footprint ('overlap'), (b) far from it ('clear'),
(c) tiny crops fully inside a band ('tiny').  Retention = fraction of the crop's ink-core pixels that are non-white in the output."""
import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
from PIL import Image
cv2.setNumThreads(2)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B0 = ld(1), ld(2)
rng = np.random.default_rng(5)
FAINT = float(os.environ.get('FAINT', '1.0'))
# ---- crop library from a 300 dpi corpus page (blue/purple/red/pencil handwriting) and the user's notes
C = np.array(Image.open(_ROOT + '/research/r2_cleaning/work/pages/38d5b2f0/pg-001.jpg').convert('RGB'))
rc = duplex.native_density(C, 300.); dC = np.maximum(np.maximum(rc[0], rc[1]), rc[2])
def crops(rr, d, w, h, n, kind, rng):
    out = []; H, W = d.shape
    for _ in range(4000):
        y = int(rng.integers(0, H - h)); x = int(rng.integers(0, W - w))
        dd = d[y:y+h, x:x+w]; cov = (dd > 0.25).mean()
        if cov < 0.04: continue
        ink = dd > 0.25
        m = [rr[c][y:y+h, x:x+w][ink].mean() for c in range(3)]
        chroma = max(m) - min(m)
        if kind == 'blue' and not (chroma > 0.25 and m[0] > m[2] + .1): continue
        if kind == 'red' and not (chroma > 0.25 and m[2] > m[0] + .1): continue
        if kind == 'neutral' and not (chroma < 0.08): continue
        out.append((x, y, w, h))
        if len(out) >= n: break
    return out
def paste(Bimg, C, rcC, dcC, box, at):
    x, y, w, h = box; px, py = at
    crop = C[y:y+h, x:x+w].astype(np.float32); 
    # transmittance of the crop relative to its local paper (approx 1-r): from flat-field planes
    tr = np.stack([1 - FAINT * rcC[c][y:y+h, x:x+w] for c in range(3)], -1)
    P = Bimg.astype(np.float32).copy()
    P[py:py+h, px:px+w] *= tr
    core = np.zeros(Bimg.shape[:2], bool); core[py:py+h, px:px+w] = FAINT * dcC[y:y+h, x:x+w] > 0.25 * min(FAINT, 1.0)
    return np.clip(P + .5, 0, 255).astype(np.uint8), core
# band footprint from the real registration
mA = duplex.ink_maps(A, 300.); mB = duplex.ink_maps(B0, 300.)
f0 = duplex.register_pair(mA, mB); reg0 = f0['reg']
rA = duplex.source_in_B_frame(A, reg0, B0.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
foot = cv2.GaussianBlur(np.minimum(cA, .6), (0, 0), 5) > 0.12
bl = (cA > 0.3) & (rA[0] > rA[2] + 0.15)       # blue/purple marker bands
ys, xs = np.nonzero(bl[::3, ::3]); band_pts = np.stack([xs * 3, ys * 3], 1)
footd = cv2.dilate(foot.astype(np.uint8), np.ones((91, 91), np.uint8)) > 0
H, W = B0.shape[:2]
# keep clear of the existing real notes of B for the 'clear' class and of other writing: use B's own strong ink
rB = duplex.native_density(B0, 300.); dB0 = np.maximum(np.maximum(rB[0], rB[1]), rB[2])
busy = cv2.dilate((dB0 > 0.15).astype(np.uint8), np.ones((61, 61), np.uint8)) > 0
lib = {k: crops(rc, dC, 240, 100, 8, k, rng) for k in ('blue', 'red', 'neutral')}
print({k: len(v) for k, v in lib.items()})
trials = []
def run(Bimg, core, cls, kind, at=None, wh=None):
    mBp = duplex.ink_maps(Bimg, 300.)
    f = duplex.register_pair(mA, mBp)
    out = duplex._clean.clean_page(Bimg, 300.)
    if f.get('reg') is None: return None
    s = duplex.bleed_suppression(A, Bimg, f['reg'])
    new = duplex.apply_suppression(out, s)
    nw = lambda o: (o.min(-1) < 250)
    if at is not None and cls == 'overlap' and FAINT == 1.0:
        px, py = at; w, h = wh; pad = 40
        sl = (slice(max(py - pad, 0), py + h + pad), slice(max(px - pad, 0), px + w + pad))
        Image.fromarray(np.hstack([Bimg[sl], out[sl], new[sl]])).save(f'work/v/synthex_{kind}_{px}_{py}.png')
    if at is not None and nw(out)[core].mean() - nw(new)[core].mean() > 0.5:
        px, py = at; w, h = wh; pad = 40
        sl = (slice(max(py - pad, 0), py + h + pad), slice(max(px - pad, 0), px + w + pad))
        Image.fromarray(np.hstack([Bimg[sl], out[sl], new[sl]])).save(f'work/v/synthfail_{cls}_{kind.replace('/','-')}_{px}_{py}.png')
    return dict(cls=cls, kind=kind, at=at, core=int(core.sum()), ret_cur=float(nw(out)[core].mean()), ret_new=float(nw(new)[core].mean()),
                lost=int((nw(out) & ~nw(new))[core].sum()))
for kind, boxes in lib.items():
    for box in boxes:
        x, y, w, h = box
        # overlap: centre the crop on a random marker-band pixel
        for _ in range(60):
            bx, by = band_pts[rng.integers(len(band_pts))]
            px, py = int(bx - w / 2), int(by - h / 2)
            if px < 0 or py < 0 or px + w > W or py + h > H: continue
            if (busy[py:py+h, px:px+w] & ~footd[py:py+h, px:px+w]).mean() > 0.02: continue   # do not collide with other writing
            break
        Bp, core = paste(B0, C, rc, dC, box, (px, py))
        r = run(Bp, core, 'overlap', kind, (px, py), (w, h)); r and trials.append(r)
        # clear placement: far from footprint and from other writing
        for _ in range(200):
            px, py = int(rng.integers(0, W - w)), int(rng.integers(100, H - h - 100))
            if not footd[py:py+h, px:px+w].any() and not busy[py:py+h, px:px+w].any(): break
        Bp, core = paste(B0, C, rc, dC, box, (px, py))
        r = run(Bp, core, 'clear', kind); r and trials.append(r)
# tiny crops (single glyph, ~0.1 in) fully inside a band
tl = crops(rc, dC, 36, 36, 10, 'blue', rng) + crops(rc, dC, 36, 36, 10, 'neutral', rng)
for box in tl:
    x, y, w, h = box
    bx, by = band_pts[rng.integers(len(band_pts))]
    px, py = int(bx - w / 2), int(by - h / 2)
    Bp, core = paste(B0, C, rc, dC, box, (px, py))
    r = run(Bp, core, 'tiny_inside_band', 'blue/neutral', (px, py), (w, h)); r and trials.append(r)
json.dump(trials, open('synth_results%s.json' % ('' if FAINT == 1.0 else '_faint%.1f' % FAINT), 'w'), indent=1)
import collections
for cls in ('clear', 'overlap', 'tiny_inside_band'):
    for kind in ('blue', 'red', 'neutral', 'blue/neutral'):
        T = [t for t in trials if t['cls'] == cls and t['kind'] == kind]
        if T: print(cls, kind, len(T), 'retention current %.4f new %.4f  min new %.3f  lost px %d of %d' % (np.mean([t['ret_cur'] for t in T]), np.mean([t['ret_new'] for t in T]), min(t['ret_new'] for t in T), sum(t['lost'] for t in T), sum(t['core'] for t in T)))
