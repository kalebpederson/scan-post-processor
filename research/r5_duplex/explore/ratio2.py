import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B = ld(1), ld(2)
f = duplex.register_pair(duplex.ink_maps(A, 300.), duplex.ink_maps(B, 300.)); reg = f['reg']
P = dict(duplex.SUPP, blur_in=0.015, foot_thr=0.06, tol_in=0.03)
s = duplex.bleed_suppression(A, B, reg, P)
rA = duplex.source_in_B_frame(A, reg, B.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
r = duplex.native_density(B, 300.); d = np.maximum(np.maximum(r[0], r[1]), r[2])
m = (s > .5) & (d > 0.05)
for sig in (0.01, 0.02, 0.03):
    S = cv2.GaussianBlur(np.minimum(cA, .6), (0, 0), sig * 300)
    q = d[m] / np.maximum(S[m], 0.02)
    print('sigma', sig, 'd/S pct 50/90/95/99/99.9', np.percentile(q, [50, 90, 95, 99, 99.9]).round(2), 'S median', np.median(S[m]).round(2))
