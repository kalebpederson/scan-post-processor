import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R2 = _ROOT + '/research/r2_cleaning'
import duplex
sys.path.insert(0, R2); cwd = os.getcwd(); os.chdir(R2)
from bleedlib import accepted_pairs, bleed_masks
from common import load_rgb
os.chdir(cwd)
key = sys.argv[1]
P = [p for p in accepted_pairs() if f"{p['src'][-30:]}:{p['front']}<-{p['back']}" == key]
p = P[0]; dpi = p['dpi']
M = load_rgb(p['front_file']); S = load_rgb(p['back_file'])
mM = duplex.ink_maps(M, dpi); mS = duplex.ink_maps(S, dpi)
f = duplex.register_pair(mS, mM)
out = duplex._clean.clean_page(M, dpi)
s, dbg = duplex.bleed_suppression(S, M, f['reg'], debug=True)
new = duplex.apply_suppression(out, s)
m = bleed_masks(dict(p), M)
vis = m['visible']
surv = (new.min(-1) < 250) & vis
ov = new.copy(); ov[surv] = [255, 0, 0]
sm = lambda x: cv2.resize(x, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
src_m = np.ascontiguousarray(S[:, ::-1])
rem=(out.min(-1)<250)&(new.min(-1)>=250); ov2=out.copy(); ov2[rem]=[255,0,0]
cv2.imwrite(sys.argv[2], np.hstack([sm(np.ascontiguousarray(S[:, ::-1])), sm(ov2)])[:, :, ::-1])
print(f['ncc'])
