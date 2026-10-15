import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B = ld(1), ld(2)
f = duplex.register_pair(duplex.ink_maps(A, 300.), duplex.ink_maps(B, 300.)); reg = f['reg']
P = dict(duplex.SUPP, mag_ratio=0.95)
s, dbg = duplex.bleed_suppression(A, B, reg, P, debug=True)
y0, y1, x0, x1 = dbg['roi']; sr = s[y0:y1, x0:x1]
d = dbg['d']; m = (d > 0.15) & (sr < 0.5) & dbg['Fd']
print('remaining ink px in footprint', m.sum())
for k in ('seeds', 'mismatch', 'toodark', 'prot'):
    print(k, int((m & dbg[k]).sum()))
print('mismatch only (not toodark)', int((m & dbg['mismatch'] & ~dbg['toodark']).sum()), ' cos<0.3 :', int((m & (dbg['cosp'] < 0.3) & (d > .12)).sum()), 'neutral', int((m & (dbg['ch'] < .3 * d)).sum()))
print('prot only by growth', int((m & dbg['prot'] & ~dbg['seeds']).sum()))
rA = duplex.source_in_B_frame(A, reg, B.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
bl = (cv2.dilate(((cA > 0.3) & (rA[0] > rA[2] + 0.15)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0)[y0:y1, x0:x1]
m2 = m & bl
print('in marker bands remaining', int(m2.sum()))
for k in ('seeds', 'mismatch', 'toodark', 'prot'): print(' ', k, int((m2 & dbg[k]).sum()))
print(' growth-only', int((m2 & dbg['prot'] & ~dbg['seeds']).sum()), ' not in Fd?', int((m2 & ~dbg['Fd']).sum()))
print(' mismatch&~toodark', int((m2 & dbg['mismatch'] & ~dbg['toodark']).sum()), ' neutral', int((m2 & (dbg['ch'] < .3 * d)).sum()))
