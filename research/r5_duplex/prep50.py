"""50 ppi per-channel ink densities r_c = clip(1 - rgb_c/bg_c, 0, 1) (uint8 x255, R2 'ref_background'), for the cached corpus pages
(research/r4_geometry/cache, 100 ppi RGB) and for the Brother stack (300 ppi TIFF, work/)."""
import sys, json, os, numpy as np, cv2
from multiprocessing import Pool
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
sys.path.insert(0, _ROOT + '/research/r2_cleaning')
cv2.setNumThreads(1)
from common import darkness
R4 = _ROOT + '/research/r4_geometry/'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE + '/work/r50/'; os.makedirs(OUT, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
def r50_from_rgb100(im):
    _, n = darkness(np.ascontiguousarray(im), 100)
    r = np.clip(1 - np.minimum(n, 1.0), 0, 1)
    r = cv2.resize(r, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return (r * 255 + .5).astype(np.uint8)
def f(a):
    i, e = a
    im = cv2.imread(R4 + e['fn'])[:, :, ::-1]
    np.save(f'{OUT}{i:04d}.npy', r50_from_rgb100(im)); return i
def fb(i):
    from PIL import Image
    rgb = np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')); h, w = rgb.shape[:2]
    s = cv2.resize(rgb, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    np.save(f'{OUT}b{i:03d}.npy', r50_from_rgb100(s)); return i
if __name__ == '__main__':
    idx = json.load(open(R4 + 'index.json'))
    with Pool(16) as p:
        p.map(f, list(enumerate(idx)), chunksize=8); p.map(fb, range(1, 23))
