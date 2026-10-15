import sys, json, random, numpy as np, os
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from duplex import InkMaps
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
def L(i):
    n = f'work/r50b/{i}.npy' if isinstance(i, str) else f'work/r50b/{i:04d}.npy'
    return InkMaps(np.load(n).astype(np.float32) / 255, None, 100)
def desc(m):
    return dict(chroma_px=int((m.c > 0.15).sum()), ink_px=int((m.d > 0.25).sum()), n=int(m.d.size))
def f(t):
    i, j, kind = t
    A, B = L(i), L(j)
    if abs(A.d.shape[1]-B.d.shape[1]) > 0.02*A.d.shape[1] or abs(A.d.shape[0]-B.d.shape[0]) > 0.05*A.d.shape[0]: return None
    a, b = duplex.coarse(A, B)
    return dict(i=i, j=j, kind=kind, a=a, b=b, dA=desc(A), dB=desc(B))
if __name__ == '__main__':
    docs = {}
    for k, e in enumerate(idx): docs.setdefault(e['k'], []).append(k)
    T = []
    for k, ps in docs.items():
        for u in range(len(ps) - 1): T.append((ps[u], ps[u+1], 'adj%d' % (u % 2)))
    rnd = random.Random(1)
    for k, ps in docs.items():
        pr = [(ps[u], ps[v]) for u in range(len(ps)) for v in range(u + 2, len(ps))]
        if len(pr) > 400: pr = rnd.sample(pr, 400)
        T += [(a, b, 'far') for a, b in pr]
    ncross = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    c = 0
    while c < ncross:
        a, b = rnd.sample(range(len(idx)), 2)
        if idx[a]['k'] != idx[b]['k']: T.append((a, b, 'cross')); c += 1
    print(len(T), flush=True)
    with Pool(16) as p: R = p.map(f, T, chunksize=32)
    R = [r for r in R if r and r['a'] and r['b']]
    json.dump(R, open('work/coarse.json', 'w')); print(len(R))
