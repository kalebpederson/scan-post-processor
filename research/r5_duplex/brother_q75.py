import sys, os, io, numpy as np, cv2, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
def ld(i):
    im = Image.open(f'{BR}page{i:03d}.tif').convert('RGB'); b = io.BytesIO(); im.save(b, 'JPEG', quality=75, subsampling=2); b.seek(0)
    return np.ascontiguousarray(np.array(Image.open(b).convert('RGB')))
A, B = ld(1), ld(2)
mA, mB = duplex.ink_maps(A, 300.), duplex.ink_maps(B, 300.)
f = duplex.register_pair(mA, mB); ok = duplex.accept(f, mA, mB)
print('q75 sheet1: gap %.2f/%.2f ncc %.2f agree %s accept %s' % (f['ga'], f['gb'], f['ncc'], f['agree_in'], ok))
out = duplex._clean.clean_page(B, 300.); s = duplex.bleed_suppression(A, B, f['reg']); new = duplex.apply_suppression(out, s)
rA = duplex.source_in_B_frame(A, f['reg'], B.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
bl = cv2.dilate(((cA > 0.3) & (rA[0] > rA[2] + 0.15)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
g = lambda o: 255 - o.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
print('band px >10gl: current', int((g(out)[bl] > 10).sum()), 'new', int((g(new)[bl] > 10).sum()), ' >40gl: current', int((g(out)[bl] > 40).sum()), 'new', int((g(new)[bl] > 40).sum()))
