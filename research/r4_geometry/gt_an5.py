import json,numpy as np,cv2
from methods import METHODS
R=json.load(open('gt_results.json'));idx=json.load(open('index.json'))
N=list(R[0]['res'])
cand=sorted(set(r['i'] for r in R))
base={}
for i in cand:
    g=cv2.imread(idx[i]['fn'],0); base[i]={n:METHODS[n](g,100)[0] for n in N if n!='hough'}
json.dump({str(k):v for k,v in base.items()},open('gt_base.json','w'))
a=np.array([r['a'] for r in R])
print('baseline (unrotated GT page) |est| mean per method:',{n:round(float(np.mean([abs(base[i][n]) for i in cand])),3) for n in N if n!='hough'})
for n in N:
    if n=='hough': continue
    rel=np.array([r['res'][n][0]-r['a']-base[r['i']][n] for r in R]);ar=np.abs(rel)
    print(f"{n:8s} self-consistency: mean {ar.mean():.3f} p95 {np.percentile(ar,95):.3f} max {ar.max():.2f} fail>0.3 {(ar>0.3).mean()*100:.1f}%")
