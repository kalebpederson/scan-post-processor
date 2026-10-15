import sys, json, numpy as np, os
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from run_coarse import L, desc, idx
C = json.load(open('work/coarse.json'))
def g(d): return 0 if d['border'] else d['peak']-d['second']
def f(r):
    A, B = L(r['i']), L(r['j'])
    A.shape_native = (A.d.shape[0]*2, A.d.shape[1]*2); B.shape_native = A.shape_native
    ft = duplex.register_pair(A, B)
    out = dict(i=r['i'], j=r['j'], kind=r['kind'], ga=ft['ga'], gb=ft['gb'], a=ft['a'], b=ft['b'], dA=r['dA'], dB=r['dB'])
    for k in ('best', 'ncc', 'rho', 'agree_in', 'fine'):
        if k in ft: out[k] = ft[k]
    if 'M_ab' in ft: out['lin'] = ft['M_ab'][:, :2].tolist()
    return out
if __name__ == '__main__':
    cand = [r for r in C if max(g(r['a']), g(r['b'])) >= 0.05]
    print(len(cand), flush=True)
    with Pool(16) as p: R = p.map(f, cand, chunksize=4)
    json.dump(R, open('work/fine.json', 'w'))
