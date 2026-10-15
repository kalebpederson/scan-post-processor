import json, numpy as np, collections
R = json.load(open('work/det_scores.json'))
def best(r, key): return max(r['a'][key], r['b'][key])
for kind in ['adj0','adj1','far','cross']:
    rs = [r for r in R if r['kind']==kind]
    z = np.array([max(r['a']['z'] if not r['a']['border'] else 0, r['b']['z'] if not r['b']['border'] else 0) for r in rs])
    print(kind, len(rs), 'z pcts 50/90/99/99.9/max', np.percentile(z,[50,90,99,99.9,100]).round(1), ' n z>=7', (z>=7).sum(), ' >=10', (z>=10).sum(), '>=15',(z>=15).sum())
