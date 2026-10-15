import sys, os, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
A = np.ascontiguousarray(np.array(Image.open(BR + 'page001.tif').convert('RGB'))); B = np.ascontiguousarray(np.array(Image.open(BR + 'page002.tif').convert('RGB')))
mA = duplex.ink_maps(A, 300.); mB = duplex.ink_maps(B, 300.)
f = duplex.register_pair(mA, mB)
S, _ = duplex.predicted_bleed(A, f['reg'], B.shape[:2], 300.)
r = duplex.native_density(B, 300.); d = np.maximum(np.maximum(r[0], r[1]), r[2])
x0, y0, w, h = 300, 860, 700, 240
ov = np.stack([255 * (1 - np.clip(d[y0:y0+h, x0:x0+w] * 3, 0, 1)), 255 * (1 - np.clip(d[y0:y0+h, x0:x0+w] * 3, 0, 1)) , 255 * (1 - np.clip(d[y0:y0+h, x0:x0+w] * 3, 0, 1))], -1)
ov[..., 1] = ov[..., 1] * (1 - np.clip(S[y0:y0+h, x0:x0+w], 0, 1))
cv2.imwrite('work/br3.png', ov.astype(np.uint8)[:, :, ::-1])
print(S[y0:y0+h, x0:x0+w].max(), d[y0:y0+h, x0:x0+w].max())
