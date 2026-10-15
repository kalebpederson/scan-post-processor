import json, numpy as np, truth, collections
from an_f3 import rule, srcc
R = json.load(open('work/fine.json'))
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
acc = [r for r in R if r['kind'].startswith('adj') and rule(r)]
print('adjacent accepted', len(acc))
cnt = collections.Counter()
for r in acc: cnt[r['i']] += 1; cnt[r['j']] += 1
print('pages in 2 accepted pairs (conflicts):', sum(1 for v in cnt.values() if v > 1))
docs = {idx[r['i']]['pdf'] for r in acc}; print('docs with >=1 accepted:', len(docs), 'of 291 scan docs')
S = truth.r2_sheets(); D = {(r['i'], r['j']): r for r in R}
ok = 0
for (src, a, b), (i, j) in S.items():
    r = D.get((i, j)) or D.get((j, i)); good = bool(r and rule(r)); ok += good
    print(src[-32:], a, b, 'ACCEPT' if good else 'reject', None if r is None else 'g %.2f %.2f ncc %s' % (r['ga'], r['gb'], r.get('ncc') and round(r['ncc'], 2)))
print('R2 sheets accepted', ok, 'of', len(S))
