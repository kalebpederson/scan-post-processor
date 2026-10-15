import json, numpy as np, truth
R = json.load(open('work/det_scores4.json'))
def feat(r, v='c_c'):
    a, b = r['v'][v]
    ga = a['peak'] - a['second'] if not a['border'] else 0; gb = b['peak'] - b['second'] if not b['border'] else 0
    return ga, gb
for kind in ['adj0','adj1','far','cross']:
    rs = [r for r in R if r['kind'] == kind]
    g = np.array([max(feat(r)) for r in rs]); gm = np.array([min(feat(r)) for r in rs])
    print(kind, len(rs), 'maxgap pct 50/90/99/max', np.percentile(g,[50,90,99,100]).round(3), ' >=.05:',(g>=.05).sum(),' >=.08:',(g>=.08).sum(),' >=.1:',(g>=.1).sum(),' >=.15',(g>=.15).sum(), '| both>=.05',(gm>=.05).sum(),' both>=.08',(gm>=.08).sum())
S = truth.r2_sheets(); D = {(r['i'], r['j']): r for r in R}
for (src, a, b), (i, j) in S.items():
    r = D.get((i, j)) or D.get((j, i)); ga, gb = feat(r)
    print(src[-30:], a, b, r['kind'], 'gaps %.2f %.2f' % (ga, gb), 'pk %.2f %.2f' % (r['v']['c_c'][0]['peak'], r['v']['c_c'][1]['peak']))
