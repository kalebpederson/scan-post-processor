import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B = ld(1), ld(2)
f = duplex.register_pair(duplex.ink_maps(A, 300.), duplex.ink_maps(B, 300.))
s = np.load('work/br1_s.npy')
rA = duplex.source_in_B_frame(A, f['reg'], B.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2])
dAx = cv2.dilate(dA, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
r = duplex.native_density(B, 300.); d = np.maximum(np.maximum(r[0], r[1]), r[2])
m = (s > .5) & (d > 0.05)
q = d[m] / np.maximum(dAx[m], 1e-3)
print('removed px', m.sum(), 'd pct 50/90/99/max', np.percentile(d[m], [50, 90, 99, 100]).round(2), 'ratio pct 50/90/99/99.9', np.percentile(q, [50, 90, 99, 99.9]).round(2), 'dAx median', np.median(dAx[m]).round(2))
