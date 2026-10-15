import json, cv2, numpy as np
from lib import *
PSM=11
a=json.load(open(f'{WORK}/refocr_600_psm{PSM}.json')); b=json.load(open(f'{WORK}/refocr_1200_psm{PSM}.json'))
g=cv2.imread(f'{WORK}/gray_600.png',0)
used=set(); ref=[]
for w in a:
    cx,cy=w['x']+w['w']/2,w['y']+w['h']/2
    best=None
    for j,v in enumerate(b):
        if j in used or v['text']!=w['text']: continue
        vx,vy=(v['x']+v['w']/2)/2,(v['y']+v['h']/2)/2
        d=abs(vx-cx)+abs(vy-cy)
        if d<20 and (best is None or d<best[0]): best=(d,j)
    if best:
        used.add(best[1]); 
        crop=g[w['y']:w['y']+w['h'],w['x']:w['x']+w['w']]
        w=dict(w); w['dark']=float(255-crop.mean()); w['dark_p10']=float(255-np.percentile(crop,10)); w['line_id']='%d.%d.%d'%tuple(w['line']); del w['line']
        ref.append(w)
json.dump(ref,open(f'{WORK}/consensus.json','w'))
print(len(a),len(b),len(ref))
d=np.array([w['dark'] for w in ref]); h=np.array([w['h'] for w in ref]); p=np.array([w['dark_p10'] for w in ref])
print(np.percentile(d,[5,25,50,75,95]).round(1)); print(np.percentile(h,[5,25,50,75,95])); print(np.percentile(p,[5,25,50,75,95]).round(1))
import collections; print(collections.Counter(w['text'] for w in ref).most_common(8))
