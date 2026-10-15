import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
def T(f, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); r = f(); ts.append(time.perf_counter() - t)
    return np.median(ts) * 1000, r
A, B = ld(1), ld(2); C, D = ld(3), ld(4)
tm, mA = T(lambda: duplex.ink_maps(A, 300.)); mB = duplex.ink_maps(B, 300.); mC = duplex.ink_maps(C, 300.); mD = duplex.ink_maps(D, 300.)
tc, _ = T(lambda: duplex.coarse(mA, mB))
tr, f = T(lambda: duplex.register_pair(mA, mB))
tr2, _ = T(lambda: duplex.register_pair(mA, mD))        # non-pair: coarse only (below the 0.05 gap) 
ts, s = T(lambda: duplex.bleed_suppression(A, B, f['reg']), 3)
tcl, out = T(lambda: duplex._clean.clean_page(B, 300.), 3)
tap, _ = T(lambda: duplex.apply_suppression(out, s), 3)
print('ink_maps %.0f ms/page; coarse %.0f ms; register_pair (accepted sheet, coarse+ECC) %.0f ms; register_pair non-pair %.0f ms; bleed_suppression %.0f ms; apply %.0f ms; clean_page (same page, snapshot) %.0f ms' % (tm, tc, tr, tr2, ts, tap, tcl))
