import json, numpy as np
R = json.load(open('work/det_scores.json'))
def ok(d): return not d['border']
print('shift stats for z>=10 hits by kind (dx,dy in inches)')
for kind in ['adj0','adj1','far','cross']:
    rs=[r for r in R if r['kind']==kind]
    H=[(d['dx']/50,d['dy']/50) for r in rs for d in (r['a'],r['b']) if ok(d) and d['z']>=10]
    H=np.array(H)
    print(kind,len(H),'|dx|<0.6&|dy|<0.6:',int(((abs(H[:,0])<0.6)&(abs(H[:,1])<0.6)).sum()), 'dx pct',np.percentile(H[:,0],[10,50,90]).round(2))
# bidirectional consistency
def cons(r):
    a,b=r['a'],r['b']
    if a['border'] or b['border']: return 99
    return max(abs(a['dx']-b['dx']), abs(a['dy']+b['dy']))
for kind in ['adj0','adj1','far','cross']:
    rs=[r for r in R if r['kind']==kind]
    c=np.array([cons(r) for r in rs]); zm=np.array([min(r['a']['z'],r['b']['z']) for r in rs])
    for th in (2,3,5):
        print(kind,'cons<=%d'%th,int((c<=th).sum()),'  & min z>=4:',int(((c<=th)&(zm>=4)).sum()),' & min z>=6:',int(((c<=th)&(zm>=6)).sum()))
