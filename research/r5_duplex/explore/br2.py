import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
B = np.ascontiguousarray(np.array(Image.open(BR + 'page002.tif').convert('RGB')))
s = np.load('work/br1_s.npy'); out = np.load('work/br1_out.npy'); new = np.load('work/br1_new.npy')
n, lab, st, cen = cv2.connectedComponentsWithStats((cv2.dilate((s > .3).astype(np.uint8), np.ones((41, 41), np.uint8))))
for k in range(1, n):
    x, y, w, h, a = st[k]
    if a > 2000: print(k, x, y, w, h, a)
