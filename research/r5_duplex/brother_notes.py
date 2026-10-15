import sys, os, numpy as np, cv2, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
B = np.ascontiguousarray(np.array(Image.open(BR + 'page002.tif').convert('RGB'))); A = np.ascontiguousarray(np.array(Image.open(BR + 'page001.tif').convert('RGB')))
out = np.load('work/br1_out.npy'); new = np.load('work/br1_new.npy')
f = duplex.register_pair(duplex.ink_maps(A, 300.), duplex.ink_maps(B, 300.))
rA = duplex.source_in_B_frame(A, f['reg'], B.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
bands = cv2.dilate(((cA > 0.3) & (rA[0] > rA[2] + 0.15)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))) > 0      # +-0.15 in around the blue/purple bars
r = duplex.native_density(B, 300.); d = np.maximum(np.maximum(r[0], r[1]), r[2])
notes = (d > 0.15) & ~bands
chg = (out != new).any(-1)
nw = lambda o: o.min(-1) < 250
print('notes ink px (raw d>0.15, >0.15in from the marker bars):', int(notes.sum()), ' changed by the post-pass:', int((chg & notes).sum()), ' ink recall current %.5f new %.5f' % (nw(out)[notes].mean(), nw(new)[notes].mean()))
g = lambda o: (255 - o.astype(np.float32) @ np.array([.299, .587, .114], np.float32))
print('mean darkness on notes: current %.2f new %.2f' % (g(out)[notes].mean(), g(new)[notes].mean()))
json.dump(dict(notes_px=int(notes.sum()), changed=int((chg & notes).sum())), open('brother_notes.json', 'w'))
for thr in (0.25, 0.35):
    nn = (d > thr) & ~bands
    print('d>%.2f outside bars: px %d changed %d' % (thr, nn.sum(), (chg & nn).sum()))
rem = chg & notes
n, lab, st, cen = cv2.connectedComponentsWithStats(cv2.dilate(rem.astype(np.uint8), np.ones((9, 9), np.uint8)))
for k in sorted(range(1, n), key=lambda k: -st[k][4])[:8]:
    m = (lab == k) & rem; print('  changed cluster bbox', st[k][:4], 'px', int(m.sum()), 'mean d %.2f' % d[m].mean(), 'mean chroma-ratio %.2f' % ((np.maximum(np.maximum(r[0], r[1]), r[2]) - np.minimum(np.minimum(r[0], r[1]), r[2]))[m].mean() / max(d[m].mean(), 1e-3)))
