import sys, numpy as np, cv2
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
B = np.array(Image.open(BR + 'page002.tif').convert('RGB'))
out = np.load(sys.argv[1]); new = np.load(sys.argv[2])
def crop(x, y, w, h): return np.vstack([np.hstack([im[y:y+h, x:x+w] for im in (B, out, new)])])
c1 = crop(300, 860, 600, 240); c2 = crop(750, 2250, 560, 160)
W = max(c1.shape[1], c2.shape[1]); pad = lambda c: np.pad(c, ((0,0),(0,W-c.shape[1]),(0,0)), constant_values=255)
Image.fromarray(np.vstack([pad(c1), pad(c2)])).save(sys.argv[3])
