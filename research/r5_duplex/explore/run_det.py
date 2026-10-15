import sys, json, itertools, random, numpy as np
from multiprocessing import Pool
from detlib import *
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
def f(t):
    i, j, kind = t
    A, B = load(i), load(j)
    if abs(A.shape[1]-B.shape[1]) > 0.02*A.shape[1] or abs(A.shape[0]-B.shape[0]) > 0.05*A.shape[0]: return None
    a, b = pair_scores(A, B)
    return dict(i=i, j=j, kind=kind, a=a, b=b)
if __name__ == '__main__':
    docs = {}
    for k, e in enumerate(idx): docs.setdefault(e['k'], []).append(k)
    T = []
    for k, ps in docs.items():
        for u in range(len(ps) - 1): T.append((ps[u], ps[u+1], 'adj%d' % (u % 2)))   # adj0: (1,2),(3,4).. in 1-based; adj1: (2,3)...
    nadj = len(T)
    # far pairs within doc (|i-j|>=2)
    rnd = random.Random(0)
    for k, ps in docs.items():
        pr = [(ps[u], ps[v]) for u in range(len(ps)) for v in range(u + 2, len(ps))]
        if len(pr) > 150: pr = rnd.sample(pr, 150)
        T += [(a, b, 'far') for a, b in pr]
    # cross doc
    for _ in range(6000):
        a, b = rnd.sample(range(len(idx)), 2)
        if idx[a]['k'] != idx[b]['k']: T.append((a, b, 'cross'))
    print(len(T), nadj, flush=True)
    with Pool(16) as p: R = p.map(f, T, chunksize=16)
    R = [r for r in R if r]
    json.dump(R, open('work/det_scores.json', 'w'))
    print(len(R))
