import sys, os, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B = ld(1), ld(2)
t = time.time(); mA = duplex.ink_maps(A, 300.); mB = duplex.ink_maps(B, 300.); t1 = time.time() - t
t = time.time(); f = duplex.register_pair(mA, mB); t2 = time.time() - t
print('maps ms', t1 * 1000, 'register ms', t2 * 1000, f['ga'], f['gb'], f['ncc'], f['agree_in'], f['best'])
print(f['M_ab'])
reg = f['reg']
t = time.time(); outB = duplex._clean.clean_page(B, 300.); tc = time.time() - t
t = time.time(); s, dbg = duplex.bleed_suppression(A, B, reg, debug=True); ts = time.time() - t
print("clean", tc, "supp", ts, (s>.5).sum())
new = duplex.apply_suppression(outB, s)
np.save('work/br1_out.npy', outB); np.save('work/br1_new.npy', new); np.save('work/br1_s.npy', s)
cv2.imwrite('work/br1_cmp.png', np.hstack([cv2.resize(x, None, fx=.25, fy=.25, interpolation=cv2.INTER_AREA)[:, :, ::-1] for x in (B[:, ::-1], outB, new)]))
