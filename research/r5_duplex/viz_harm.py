import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R4 = _ROOT + '/research/r4_geometry/'
idx = json.load(open(R4 + 'index.json'))
rd = lambda i: np.ascontiguousarray(cv2.imread(R4 + idx[i]['fn'])[:, :, ::-1])
def show(i, j, out_png):
    """i = source page A, j = suppressed page B"""
    A, B = rd(i), rd(j)
    f = duplex.register_pair(duplex.ink_maps(A, 100.), duplex.ink_maps(B, 100.), coarse_gap_min=0.0)
    out = duplex._clean.clean_page(B, 100.)
    s = duplex.bleed_suppression(A, B, f['reg']); new = duplex.apply_suppression(out, s)
    rem = (out.min(-1) < 250) & (new.min(-1) >= 250)
    ov = out.copy(); ov[rem] = [255, 0, 0]
    srcm = np.ascontiguousarray(A[:, ::-1])
    cv2.imwrite(out_png, np.hstack([srcm, ov])[:, :, ::-1])
if __name__ == '__main__':
    H = json.load(open('harm_eval.json'))
    for tag in ('accepted',):
        rows = [(max(r['B<-A']['frac'], r['A<-B']['frac']), r) for r in H if r['tag'] == tag and r['ok']]
        rows.sort(key=lambda x: -x[0])
        for k, (fr, r) in enumerate(rows[:3]):
            side = 'B<-A' if r['B<-A']['frac'] >= r['A<-B']['frac'] else 'A<-B'
            i, j = (r['i'], r['j']) if side == 'B<-A' else (r['j'], r['i'])
            show(i, j, f'work/v/harm_{tag}_{k}.png'); print(tag, k, fr, i, j, idx[j]['pdf'][-40:], idx[j]['page'])
