"""Does any consecutive pair need a different flip (vertical flip = long-edge/short-edge turn, rot180, none)?"""
import sys, json, numpy as np, os
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from duplex import InkMaps
from run_coarse import L, idx
def g(d): return 0 if (d is None or d['border']) else d['peak'] - d['second']
def tr(m, f):
    return InkMaps(f(m.r), None, 100)
MODES = {'mirror': lambda r: r, 'vflip': lambda r: r[::-1, ::-1], 'rot180': lambda r: r[::-1, :], 'none': lambda r: r[:, ::-1]}
def f(t):
    i, j = t
    A, B = L(i), L(j); out = {}
    if abs(A.d.shape[1]-B.d.shape[1]) > 0.02*A.d.shape[1] or abs(A.d.shape[0]-B.d.shape[0]) > 0.05*A.d.shape[0]: return None
    for k, fn in MODES.items():
        a, b = duplex.coarse(A, tr(B, fn))
        out[k] = (g(a), g(b))
    return dict(i=i, j=j, **{k: v for k, v in out.items()})
if __name__ == '__main__':
    docs = {}
    for k, e in enumerate(idx): docs.setdefault(e['k'], []).append(k)
    T = [(ps[u], ps[u+1]) for ps in docs.values() for u in range(len(ps)-1)]
    with Pool(12) as p: R = [r for r in p.map(f, T, chunksize=8) if r]
    json.dump(R, open('work/flip.json', 'w'))
    for k in MODES:
        mx = np.array([max(r[k]) for r in R]); mn = np.array([min(r[k]) for r in R])
        print(k, len(R), 'max gap>=.15:', int((mx >= .15).sum()), ' >=.1:', int((mx >= .1).sum()), ' both>=.08:', int((mn >= .08).sum()))
