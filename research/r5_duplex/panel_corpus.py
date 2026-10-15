import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R2 = _ROOT + '/research/r2_cleaning'
import duplex
sys.path.insert(0, R2); cwd = os.getcwd(); os.chdir(R2)
from bleedlib import accepted_pairs
from common import load_rgb
os.chdir(cwd)
KEYS = [" Similarity and Properties.pdf:4<-3", "dary II - Quadratic Models.pdf:3<-2", "operties of Parallelograms.pdf:3<-2", "dary II - Quadratic Models.pdf:2<-3"]
W, H = 440, 280
rows = []
for key in KEYS:
    p = [p for p in accepted_pairs() if f"{p['src'][-30:]}:{p['front']}<-{p['back']}" == key][0]; dpi = p['dpi']
    M = load_rgb(p['front_file']); S = load_rgb(p['back_file'])
    f = duplex.register_pair(duplex.ink_maps(S, dpi), duplex.ink_maps(M, dpi))
    out = duplex._clean.clean_page(M, dpi); s = duplex.bleed_suppression(S, M, f['reg']); new = duplex.apply_suppression(out, s)
    rem = ((out.min(-1) < 250) & (new.min(-1) >= 250)).astype(np.float32)
    ii = cv2.boxFilter(rem, -1, (W, H), normalize=False); ii[:H // 2] = 0; ii[-H // 2:] = 0; ii[:, :W // 2] = 0; ii[:, -W // 2:] = 0
    cy, cx = np.unravel_index(ii.argmax(), ii.shape); y0, x0 = cy - H // 2, cx - W // 2
    rA = duplex.source_in_B_frame(S, f['reg'], M.shape[:2]); dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2])
    src = np.stack([(255 * (1 - np.clip(dA * 1.2, 0, 1))).astype(np.uint8)] * 3, -1)
    cols = []
    for im, t in ((M, 'raw'), (out, 'current'), (new, 'new'), (src, "A's ink, registered")):
        c = np.ascontiguousarray(im[y0:y0 + H, x0:x0 + W]); c = np.pad(c, ((20, 0), (0, 0), (0, 0)), constant_values=255)
        cv2.putText(c, t + '  ' + key.strip()[-34:], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 0, 0), 1, cv2.LINE_AA); cols.append(c)
    rows.append(np.hstack(cols)); print(key, dpi, int(rem.sum()))
cv2.imwrite('panels/corpus_pairs_crops.png', np.vstack(rows)[:, :, ::-1])
