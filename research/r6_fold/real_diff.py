"""Pixels that differ between the shipped output (scratch/cur) and the output with fold removal (scratch/new) outside the fold bands (+-0.1 in around the fold rows).
Run after real_eval.py.  Zero by construction for the pages without a detected fold."""
import json
import numpy as np
from common import *
R = json.load(open('real_results.json'))['pages']
out = {}
for p, v in R.items():
    cur = np.load(f'{SCR}/cur/p{int(p):02d}.npy'); new = np.load(f'{SCR}/new/p{int(p):02d}.npy')
    h = cur.shape[0]
    keep = np.ones(h, bool)
    for y, tier in v['folds']:
        keep[max(0, int(y * h - 0.1 * 300)):int(y * h + 0.1 * 300)] = False
    d = (cur != new).any(2)
    dd = np.abs(cur.astype(int) - new.astype(int)).max(2)
    out[p] = dict(folds=len(v['folds']), changed_px_total=int(d.sum()), changed_px_outside_bands=int(d[keep].sum()), max_abs_diff_outside=int(dd[keep].max()), changed_px_outside_gt_10gl=int((dd[keep] > 10).sum()))
    print(p, out[p])
json.dump(out, open('real_diff.json', 'w'), indent=1)
