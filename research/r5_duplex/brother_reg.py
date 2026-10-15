import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from duplex import InkMaps
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
def Lb(i): return InkMaps(np.load(f'work/r50b/b{i:03d}.npy').astype(np.float32) / 255, (3269, 2549), 300)
out = []
for i in range(1, 22):
    A, B = Lb(i), Lb(i + 1)
    A.shape_native = (A.d.shape[0] * 6, A.d.shape[1] * 6); B.shape_native = A.shape_native
    f = duplex.register_pair(A, B)
    print(i, i + 1, 'SHEET' if i % 2 else 'cross', 'ga %.2f gb %.2f' % (f['ga'], f['gb']), 'ncc', f.get('ncc') and round(f['ncc'], 3), 'rho', f.get('rho') and round(f['rho'], 3), 'agree', f.get('agree_in'), 'best', f.get('best'),
          '' if f.get('M_ab') is None else np.round(f['M_ab'], 4).tolist())
