import sys, numpy as np, cv2, os, json
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
sys.path.insert(0, _ROOT + '/research/r2_cleaning')
from common import darkness
RAW = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
OUT = 'work/brother50/'; os.makedirs(OUT, exist_ok=True)
from multiprocessing import Pool
def f(i):
    rgb = np.array(Image.open(f'{RAW}page{i:03d}.tif').convert('RGB'))
    h, w = rgb.shape[:2]
    s = cv2.resize(rgb, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    d, _ = darkness(s, 100)
    d50 = cv2.resize(d, (s.shape[1] // 2, s.shape[0] // 2), interpolation=cv2.INTER_AREA)
    np.save(f'{OUT}{i:03d}.npy', d50.astype(np.float16)); return i, rgb.shape
if __name__ == '__main__':
    with Pool(11) as p: print(p.map(f, range(1, 23)))
