import json, numpy as np, truth
R = json.load(open('work/coarse.json'))
def g(d): return 0 if d['border'] else d['peak']-d['second']
def feats(r):
    a,b=r['a'],r['b']; return g(a),g(b)
print(len(R))
for kind in ['adj0','adj1','far','cross']:
    rs=[r for r in R if r['kind']==kind]
    mx=np.array([max(feats(r)) for r in rs]); mn=np.array([min(feats(r)) for r in rs])
    print(kind,len(rs),'max>=.05/.08/.1/.15/.2:',[(mx>=t).sum() for t in (.05,.08,.1,.15,.2)],' min>=.05/.08/.1:',[(mn>=t).sum() for t in (.05,.08,.1)])
