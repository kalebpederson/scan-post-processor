import json, numpy as np
R = json.load(open('work/det_scores.json'))
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
def zz(r): return max(r['a']['z'] if not r['a']['border'] else 0, r['b']['z'] if not r['b']['border'] else 0)
for kind in ['cross','far']:
    rs = sorted([r for r in R if r['kind']==kind], key=lambda r:-zz(r))[:12]
    for r in rs:
        print(kind, round(zz(r),1), r['i'], r['j'], idx[r['i']]['pdf'][-40:], idx[r['i']]['page'], '|', idx[r['j']]['pdf'][-40:], idx[r['j']]['page'], r['a']['peak'], r['b']['peak'])
