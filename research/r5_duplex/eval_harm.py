"""(1) ink removed on ACCEPTED adjacent pairs (both directions) of the 100 ppi corpus cache; (2) harm if the gate were bypassed: forced
registration + suppression on the 300 highest-scoring REJECTED non-pairs.  Metric: cleaned-ink pixels (min channel < 250) turned white."""
import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
from multiprocessing import Pool
cv2.setNumThreads(1)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R4 = _ROOT + '/research/r4_geometry/'
idx = json.load(open(R4 + 'index.json'))
rd = lambda i: np.ascontiguousarray(cv2.imread(R4 + idx[i]['fn'])[:, :, ::-1])
def srcc(r): return r['dB']['chroma_px'] if r['best'] == 'a' else r['dA']['chroma_px']
def gate(r, Cmin=400):
    if r.get('ncc') is None: return False
    two = r.get('agree_in') is not None and r['ga'] >= .05 and r['gb'] >= .05 and r['agree_in'] <= .12 and r['ncc'] >= .2
    one = max(r['ga'], r['gb']) >= .06 and r['ncc'] >= .5
    return (two or one) and srcc(r) >= Cmin
def task(t):
    i, j, tag = t
    A, B = rd(i), rd(j)
    mA, mB = duplex.ink_maps(A, 100.), duplex.ink_maps(B, 100.)
    f = duplex.register_pair(mA, mB, coarse_gap_min=0.0)
    if f.get('reg') is None: return dict(tag=tag, i=i, j=j, ok=False)
    res = dict(tag=tag, i=i, j=j, ok=True)
    for name, X, Y, reg in (('B<-A', A, B, f['reg']), ('A<-B', B, A, duplex.PairReg(M_ab=duplex._inv(f['reg'].M_ab), dpi_A=100., dpi_B=100.))):
        out = duplex._clean.clean_page(Y, 100.)
        s = duplex.bleed_suppression(X, Y, reg)
        new = duplex.apply_suppression(out, s)
        nw = lambda o: o.min(-1) < 250
        ink = int(nw(out).sum()); rem = int((nw(out) & ~nw(new)).sum())
        res[name] = dict(ink=ink, removed=rem, frac=rem / max(ink, 1), dark_removed=int(((out.min(-1) < 120) & ~nw(new)).sum()))
    return res
if __name__ == '__main__':
    F = json.load(open('work/fine.json'))
    acc = [(r['i'], r['j'], 'accepted') for r in F if r['kind'].startswith('adj') and gate(r)]
    rej = sorted([r for r in F if r['kind'] in ('far', 'cross') and r.get('ncc') is not None and not gate(r)], key=lambda r: -max(r['ga'], r['gb']))
    # fill with random far/cross pairs so the sample is not only the top scorers
    rnd = np.random.default_rng(0)
    T = acc + [(r['i'], r['j'], 'false_top') for r in rej[:100]]
    C = json.load(open('work/coarse.json')); far = [r for r in C if r['kind'] in ('far', 'cross')]
    T += [(r['i'], r['j'], 'false_random') for r in rnd.choice(far, 200, replace=False)]
    with Pool(12) as p: R = p.map(task, T, chunksize=2)
    json.dump(R, open('harm_eval.json', 'w'))
    for tag in ('accepted', 'false_top', 'false_random'):
        v = [(r['B<-A'], r['A<-B']) for r in R if r['tag'] == tag and r['ok']]
        fr = np.array([x['frac'] for p in v for x in p]); dr = np.array([x['dark_removed'] for p in v for x in p])
        print(tag, 'registered', len(v), 'of', sum(1 for r in R if r['tag'] == tag), 'side-instances', len(fr), 'removed-ink frac mean %.5f p99 %.5f max %.5f' % (fr.mean(), np.percentile(fr, 99), fr.max()), 'any dark(<120) px removed:', int((dr > 0).sum()), 'sum', int(dr.sum()))
