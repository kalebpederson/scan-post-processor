"""Speed of the fold detector / remover, single thread (taskset + cv2.setNumThreads(1)), on the 14 pages of the test document (300 ppi, 8.3 Mpx)
and on a few corpus pages at their native resolution.  Compared with clean_page and the pipeline's blank score on the same pages."""
import json, time, os
import numpy as np, cv2
cv2.setNumThreads(1)
from common import *
import fold as F
from scanpp.clean import clean_page
from scanpp import geometry

P = F.FoldParams()
def cpu(f, n=3):
    best = 1e9
    for _ in range(n):
        t = time.process_time(); r = f(); best = min(best, time.process_time() - t)
    return best, r

if __name__ == '__main__':
    out = {}
    for p in (13, 9, 4, 1):
        raw = td_page(p)
        d = {}
        d['clean_page'], _ = cpu(lambda: clean_page(raw, 300.0), 3)
        d['blank_score'], _ = cpu(lambda: geometry.blank_score(raw, 300.0), 3)
        d['stage1_detect_from_native'], cs = cpu(lambda: F.line_candidates(raw, 300.0, P), 3)
        g100 = cv2.resize(raw, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA)
        d['stage1_on_100ppi_rgb'], _ = cpu(lambda: F.line_candidates(g100, 100.0, P), 3)
        d['downscale_to_100ppi'], _ = cpu(lambda: cv2.resize(raw, None, fx=1 / 3, fy=1 / 3, interpolation=cv2.INTER_AREA), 3)
        folds = F.decide(cs[0], cs[1][0], cs[1][1], P, None) or [F.Fold(.5, 0, 0, 1, (1, 1), 9)]
        d['native_check_and_track'], _ = cpu(lambda: [F._fold_geometry(raw, 300.0, f, P) for f in folds], 3)
        d['fold_mask'], mask = cpu(lambda: F.fold_mask(raw, 300.0, folds, P), 3)
        d['remove_folds_total'], _ = cpu(lambda: F.remove_folds(raw, 300.0, folds, P), 3)
        d['n_folds_timed'] = len(folds)
        out[f'p{p}'] = {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
        print(p, out[f'p{p}'], flush=True)
    json.dump(out, open('speed.json', 'w'), indent=1)
