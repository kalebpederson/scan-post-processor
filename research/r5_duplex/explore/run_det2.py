import sys, json, numpy as np
from multiprocessing import Pool
from detlib import *
R0 = json.load(open('work/det_scores.json'))
def f(r):
    A, B = load(r['i']), load(r['j'])
    A, B = common(A, B)
    out = dict(r)
    for key, X, Y in (('a', A, B), ('b', B, A)):
        d = r[key]
        if d['z'] < 4 or d['peak'] < 0.06: continue
        out[key] = dict(d, **extra_feats(X, Y[:, ::-1], d['dx'], d['dy']))
    return out
if __name__ == '__main__':
    with Pool(16) as p: R = p.map(f, R0, chunksize=32)
    json.dump(R, open('work/det_scores2.json', 'w'))
