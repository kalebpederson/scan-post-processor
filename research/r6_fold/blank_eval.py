"""Blank-page safety of the fold removal on the whole corpus (R4 100-ppi cache, edge-cleaned like the pipeline).

For every page: blank_score as shipped (s0), blank_score after the detector's removal (s1; the detector fires on no corpus page, so s1 == s0), and a
STRESS value s2: the removal forced (as if the detector had fired) at all three customary rows 1/3, 1/2, 2/3 with angle 0, tracked and masked
like a real removal.  The question is whether a page with real content could be dropped (blank_score < 15) by a wrong removal.
Writes blank_eval.json (aggregates + the 20 lowest non-blank scores before / after the stress).
"""
import json
from multiprocessing import Pool
import numpy as np
from common import *
import fold as F
from scanpp import geometry

P = F.FoldParams()
THR = 15.0

def work(i):
    try:
        rgb, r = corpus_page(i)
        img, info, gray, wd = geometry._clean_edges_g(rgb.copy(), 100.0, inplace=True)
        s0 = geometry.blank_score(img, 100.0)
        cs, (wi, hi) = F.line_candidates(img, 100.0, P)
        folds = F.decide(cs, wi, hi, P, None)
        if folds:
            rm = F.remove_folds(img, 100.0, folds, P)
            s1 = geometry.blank_score(rm, 100.0)
        else:
            s1 = s0
        forced = [F.Fold(y, 0.0, 0.0, 1.0, (1, 1), 99) for y in (1 / 3, 1 / 2, 2 / 3)]
        rm2 = F.remove_folds(img, 100.0, forced, P)
        s2 = geometry.blank_score(rm2, 100.0)
        return i, (float(s0), float(s1), float(s2), len(folds))
    except Exception as e:
        return i, repr(e)

if __name__ == '__main__':
    with Pool(14) as p:
        res = dict(p.map(work, range(len(index())), chunksize=8))
    bad = {k: v for k, v in res.items() if isinstance(v, str)}
    ok = {k: v for k, v in res.items() if not isinstance(v, str)}
    s0 = np.array([v[0] for v in ok.values()]); s1 = np.array([v[1] for v in ok.values()]); s2 = np.array([v[2] for v in ok.values()]); nf = np.array([v[3] for v in ok.values()])
    keys = np.array(list(ok.keys()))
    nonblank = s0 >= THR
    out = dict(pages=len(res), errors=len(bad), detector_fired_pages=int((nf > 0).sum()), blank_pages=int((~nonblank).sum()),
               nonblank_min_s0=float(s0[nonblank].min()), nonblank_min_s1=float(s1[nonblank].min()), nonblank_min_s2_stress=float(s2[nonblank].min()),
               changed_s0_to_s1=int((np.abs(s1 - s0) > 1e-6).sum()), new_drops_s1=int(((s1 < THR) & nonblank).sum()), new_drops_stress=int(((s2 < THR) & nonblank).sum()))
    o = np.argsort(np.where(nonblank, s0, 1e18))[:20]
    out['lowest20_nonblank'] = [dict(page=int(keys[j]), s0=round(float(s0[j]), 1), s1=round(float(s1[j]), 1), s2_stress=round(float(s2[j]), 1)) for j in o]
    ratio = np.where(nonblank, s2 / np.maximum(s0, 1e-9), 1.0)
    out['stress_ratio_min'] = float(ratio.min())
    out['stress_ratio_p1'] = float(np.percentile(ratio[nonblank], 1))
    o2 = np.argsort(np.where(nonblank, s2, 1e18))[:20]
    out['lowest20_after_stress'] = [dict(page=int(keys[j]), s0=round(float(s0[j]), 1), s2_stress=round(float(s2[j]), 1)) for j in o2]
    json.dump(out, open('blank_eval.json', 'w'), indent=1)
    print({k: v for k, v in out.items() if not k.startswith('lowest')})
    print(out['lowest20_nonblank'][:8])
