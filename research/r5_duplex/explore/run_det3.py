import sys, json, numpy as np
from multiprocessing import Pool
from detlib2 import *
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
R0 = json.load(open('work/det_scores.json'))
def f(r):
    A, B = load_r(r['i']), load_r(r['j'])
    if abs(A.shape[1]-B.shape[1]) > 0.02*A.shape[1] or abs(A.shape[0]-B.shape[0]) > 0.05*A.shape[0]: return None
    return dict(i=r['i'], j=r['j'], kind=r['kind'], v=variants2(A, B))
if __name__ == '__main__':
    with Pool(16) as p: R = p.map(f, R0, chunksize=16)
    json.dump([r for r in R if r], open('work/det_scores4.json', 'w'))
