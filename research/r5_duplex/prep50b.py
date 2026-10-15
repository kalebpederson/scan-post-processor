"""Recompute 50 ppi maps with duplex.ink_maps from the 100 ppi corpus cache and Brother TIFFs -> work/r50b/*.npy (uint8 r x255, 3ch)."""
import sys, json, os, numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
cv2.setNumThreads(1)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R4 = _ROOT + '/research/r4_geometry/'
OUT = os.path.dirname(os.path.abspath(__file__)) + '/work/r50b/'; os.makedirs(OUT, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
def f(a):
    i, e = a
    im = np.ascontiguousarray(cv2.imread(R4 + e['fn'])[:, :, ::-1])
    m = duplex.ink_maps(im, 100.0); np.save(f'{OUT}{i:04d}.npy', (m.r * 255 + .5).astype(np.uint8)); return i
def fb(i):
    rgb = np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB'))
    m = duplex.ink_maps(rgb, 300.0); np.save(f'{OUT}b{i:03d}.npy', (m.r * 255 + .5).astype(np.uint8)); return i
if __name__ == '__main__':
    idx = json.load(open(R4 + 'index.json'))
    with Pool(16) as p:
        p.map(f, list(enumerate(idx)), chunksize=8); p.map(fb, range(1, 23))
