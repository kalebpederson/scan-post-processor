import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
bands = cv2.dilate(((cA > 0.3) & (rA[0] > rA[2] + 0.15)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (91, 91))) > 0
r = duplex.native_density(B, 300.); d = np.maximum(np.maximum(r[0], r[1]), r[2])
g = lambda o: 255 - o.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
notes = (d > 0.25) & ~bands
dot = np.zeros(d.shape, bool); dot[630:662, 1207:1243] = d[630:662, 1207:1243] > 0.25
for cm in (0.6, 0.7, 0.8, 0.9, 5.0):
    P = dict(duplex.SUPP, chroma_mag=cm)
    s = duplex.bleed_suppression(A, B, reg, P); new = duplex.apply_suppression(out, s)
    chg = (out != new).any(-1)
    print('chroma_mag', cm, 'bands >10gl:', int((g(new)[bl] > 10).sum()), '>40:', int((g(new)[bl] > 40).sum()), ' notes(d>.25 outside bars) changed:', int((chg & notes).sum()), ' red dot px changed', int((chg & dot).sum()), 'of', int(dot.sum()))
