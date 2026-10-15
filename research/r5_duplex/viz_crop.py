import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R2 = _ROOT + '/research/r2_cleaning'
import duplex
sys.path.insert(0, R2); cwd = os.getcwd(); os.chdir(R2)
from bleedlib import accepted_pairs
from common import load_rgb
os.chdir(cwd)
key = sys.argv[1]; x, y, w, h = map(int, sys.argv[3:7]); sc = int(sys.argv[7]) if len(sys.argv) > 7 else 1
p = [p for p in accepted_pairs() if f"{p['src'][-30:]}:{p['front']}<-{p['back']}" == key][0]; dpi = p['dpi']
M = load_rgb(p['front_file']); S = load_rgb(p['back_file'])
f = duplex.register_pair(duplex.ink_maps(S, dpi), duplex.ink_maps(M, dpi))
out = duplex._clean.clean_page(M, dpi); s = duplex.bleed_suppression(S, M, f['reg']); new = duplex.apply_suppression(out, s)
rA = duplex.source_in_B_frame(S, f['reg'], M.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2])
src = (255 * (1 - np.clip(dA, 0, 1))).astype(np.uint8); src = np.stack([src] * 3, -1)
c = lambda im: cv2.resize(im[y:y+h, x:x+w], None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(sys.argv[2], np.hstack([c(M), c(out), c(new), c(src)])[:, :, ::-1])
