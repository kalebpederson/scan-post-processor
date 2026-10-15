import sys, os, json, numpy as np, cv2, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B = ld(1), ld(2)
f = duplex.register_pair(duplex.ink_maps(A, 300.), duplex.ink_maps(B, 300.)); reg = f['reg']
out = duplex._clean.clean_page(B, 300.)
rA = duplex.source_in_B_frame(A, reg, B.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
bl = cv2.dilate(((cA > 0.3) & (rA[0] > rA[2] + 0.15)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
r = duplex.native_density(B, 300.); dB = np.maximum(np.maximum(r[0], r[1]), r[2])
for blur, thr, op, tol in ((0.01, 0.15, 0.05, 0.05), (0.01, 0.15, 0.04, 0.05), (0.01, 0.12, 0.05, 0.04), (0.01, 0.2, 0.05, 0.05), (0.01, 0.15, 0.06, 0.05), (0.01,0.15,0.05,0.07)):
    P = dict(duplex.SUPP, blur_in=blur, foot_thr=thr, open_in=op, tol_in=tol)
    s = duplex.bleed_suppression(A, B, reg, P); new = duplex.apply_suppression(out, s)
    g = 255 - new.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
    print('blur', blur, 'thr', thr, 'open', op, 'tol', tol, 'after >10gl', int((g[bl] > 10).sum()), '>40gl', int((g[bl] > 40).sum()))
