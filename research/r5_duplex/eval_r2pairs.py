"""Bleed survival on R2's 35 verified directed instances, before/after the duplex post-pass (detector gate vs forced registration)."""
import sys, os, json, time, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R2 = _ROOT + '/research/r2_cleaning'
import duplex
sys.path.insert(0, R2); os.chdir(R2)
from bleedlib import accepted_pairs, bleed_masks
from common import load_rgb
from evaluate import dE_white
os.chdir(HERE)
from multiprocessing import Pool
cv2.setNumThreads(1)
sys.path.insert(0, HERE)

def gate(f, srcchroma, Cmin=400):
    if f.get('ncc') is None: return False
    two = f.get('agree_in') is not None and f['ga'] >= .05 and f['gb'] >= .05 and f['agree_in'] <= .12 and f['ncc'] >= .2
    one = max(f['ga'], f['gb']) >= .15 and f['ncc'] >= .35
    return (two or one) and srcchroma >= Cmin
def one(p):
    dpi = p['dpi']; measured = load_rgb(p['front_file']); source = load_rgb(p['back_file'])
    mM = duplex.ink_maps(measured, dpi); mS = duplex.ink_maps(source, dpi)
    f = duplex.register_pair(mS, mM)          # A = source, B = measured
    # chroma px at 100ppi-equivalent area scale used by the gate (gate was calibrated on 50 ppi maps of the 100 ppi cache): same map size
    acc = duplex.accept(f, mS, mM)
    out = duplex._clean.clean_page(measured, dpi)
    res = dict(key=f"{p['src'][-30:]}:{p['front']}<-{p['back']}", accepted=bool(acc), ga=f['ga'], gb=f['gb'], ncc=f.get('ncc'))
    m = bleed_masks(dict(p), measured)
    vis = m['visible']
    def surv(o): return float((o.min(-1) < 250)[vis].mean())
    res['nvis'] = int(vis.sum()); res['before'] = surv(out)
    if f.get('reg') is not None:
        t = time.perf_counter(); s = duplex.bleed_suppression(source, measured, f['reg']); res['t_supp'] = time.perf_counter() - t
        new = duplex.apply_suppression(out, s)
        res['after_forced'] = surv(new); res['after'] = res['after_forced'] if acc else res['before']
        res['removed_px'] = int(((out.min(-1) < 250) & (new.min(-1) >= 250)).sum()); res['ink_px'] = int((out.min(-1) < 250).sum()); res['dark_removed_px'] = int(((out.min(-1) < 110) & (new.min(-1) >= 250)).sum()); res['dark_px'] = int((out.min(-1) < 110).sum())
    else:
        res['after_forced'] = res['after'] = res['before']; res['removed_px'] = 0; res['dark_removed_px'] = 0; res['dark_px'] = 1; res['ink_px'] = int((out.min(-1) < 250).sum())
    return res
if __name__ == '__main__':
    P = accepted_pairs()
    with Pool(12) as pool: R = pool.map(one, P)
    json.dump(R, open('r2pairs_eval.json', 'w'), indent=1)
    nv = np.array([r['nvis'] for r in R], float)
    for k in ('before', 'after', 'after_forced'):
        v = np.array([r[k] for r in R]); print(k, 'pooled %.4f macro %.4f' % ((v * nv).sum() / nv.sum(), v.mean()))
    print('accepted', sum(r['accepted'] for r in R), 'of', len(R))
