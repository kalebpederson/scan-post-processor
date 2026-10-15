import sys, numpy as np, cv2
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
B = np.array(Image.open(BR + 'page002.tif').convert('RGB'))
out = np.load('work/br1_out.npy'); new = np.load('work/br1_new.npy')
W = 620
def row(x, y, w, h, label):
    cols = []
    for im, t in ((B, 'raw'), (out, 'current scanpp'), (new, 'new (duplex post-pass)')):
        c = np.full((h + 22, W, 3), 255, np.uint8); c[22:22 + h, :min(w, W)] = im[y:y + h, x:x + min(w, W)]
        cv2.putText(c, f'{t}  {label}', (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(c, (0, 0), (W - 1, h + 21), (180, 180, 180), 1)
        cols.append(c)
    return np.hstack(cols)
rows = [row(300, 860, 600, 240, 'blue marker bleed (y=860)'), row(560, 2040, 600, 180, 'purple bars (y=2040)'),
        row(940, 740, 620, 200, 'user note: green'), row(940, 1600, 620, 180, 'user note: 0.3 mm pencil'), row(940, 440, 620, 220, 'user note: red')]
Image.fromarray(np.vstack(rows)).save('panels/brother_sheet1_back_1to1.png')
print(np.vstack(rows).shape)
