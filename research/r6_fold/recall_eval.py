"""Detection recall on synthetic creases (real crease strips pasted over corpus pages at customary fold rows) for several FoldParams variants.

Instance: corpus page (native, edge-cleaned) x strip (p4 / p6 / p14 / p13) x row in {1/3, 1/2, 2/3} +- 0.01, tilt +-0.3 deg, with the crease's own wander.
Recall = the detector (tier A: decide(), single page) returns a fold within 0.015 of the pasted row.  The same variants are scored on the corpus
(sweep3 per variant: python detect_eval.py ablate) for false detections; this script only gives the recall side.
Usage: python recall_eval.py [n_pages_per_class]  -> recall_results.json (per variant / strip / class)
"""
import json, sys
from multiprocessing import Pool
import numpy as np
from common import *
import fold as F
from synth_eval import choose_pages, load_strip, load_wander, paste, STRIPS
from scanpp import geometry

VARIANTS = {
    'default': {},
    'obs_frac_0.3': dict(obs_frac_min=0.3),
    'obs_frac_0.2': dict(obs_frac_min=0.2),
    'cov_base_0.7': dict(cov_base=0.7),
    'pres_0.65': dict(pres_frac_min=0.65),
    'cont_0.45': dict(cont_min=0.45),
    'amp_0.015': dict(amp_min=0.015),
    'loose': dict(obs_frac_min=0.3, cov_base=0.7, pres_frac_min=0.65, cont_min=0.45, amp_min=0.015),
}


def work(args):
    idx, cls, strip, j = args
    try:
        rng = np.random.default_rng(7919 * j + 13)
        page, rec = native_page(idx)
        dpi = float(rec['dpi'])
        page = geometry._clean_edges_g(np.array(page), dpi, inplace=True)[0]
        h, w = page.shape[:2]
        yf = float(rng.choice((1 / 3, 1 / 2, 2 / 3)) + rng.uniform(-0.01, 0.01))
        tilt = float(rng.uniform(-0.3, 0.3))
        pasted, _ = paste(page, load_strip(strip, dpi), yf * h, tilt, dpi, load_wander(strip))
        out = {}
        for name, kw in VARIANTS.items():
            Pv = F.FoldParams(**kw)
            cs, (wi, hi) = F.line_candidates(pasted, dpi, Pv)
            fs = F.decide(cs, wi, hi, Pv, None)
            near = [c for c in cs if abs(c['y_frac'] - yf) <= 0.015]
            out[name] = dict(detected=any(abs(f.y_frac - yf) <= 0.015 for f in fs), other=sum(1 for f in fs if abs(f.y_frac - yf) > 0.015),
                             score=round(max([c['score'] for c in near], default=0.0), 3))
        return dict(idx=idx, cls=cls, strip=strip, y=round(yf, 3), res=out)
    except Exception as e:
        return dict(idx=idx, err=repr(e))


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    pages = choose_pages(n, seed=1)
    jobs = [(i, c, s, 4 * k + m) for k, (i, c) in enumerate(pages) for m, s in enumerate(STRIPS)]
    with Pool(12) as p:
        R = p.map(work, jobs, chunksize=2)
    R = [r for r in R if 'err' not in r]
    out = {'n_instances': len(R), 'variants': {}}
    for name in VARIANTS:
        d = dict(recall=float(np.mean([r['res'][name]['detected'] for r in R])), other_detections=int(sum(r['res'][name]['other'] for r in R)))
        for key, vals in (('strip', STRIPS), ('cls', ('text', 'colour', 'faint'))):
            d['by_' + key] = {v: round(float(np.mean([r['res'][name]['detected'] for r in R if r[key] == v])), 3) for v in vals if any(r[key] == v for r in R)}
        out['variants'][name] = d
    json.dump(out, open('recall_results.json', 'w'), indent=1)
    json.dump(R, open(SCR + '/recall_raw.json', 'w'))
    for k, v in out['variants'].items():
        print(k, round(v['recall'], 3), 'other', v['other_detections'], v['by_strip'], v['by_cls'])
