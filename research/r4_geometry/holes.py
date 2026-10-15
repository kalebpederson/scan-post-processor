import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from edges import paper_level
cv2.setNumThreads(1)
coil=json.load(open('coil_raw.json'))
def work(a):
    i,m=a
    cl=coil[i]
    g=cv2.imread(m['fn'],0);h,w=g.shape;pl=paper_level(g)
    out={}
    for s in 'LR':
        if cl[s][0]>=0.1: out[s]=None; continue   # coil side: skip
        x0,x1=(0,int(1.3*100)) if s=='L' else (w-130,w)
        band=cv2.GaussianBlur(g[:,x0:x1],(0,0),1.5)
        c=cv2.HoughCircles(band,cv2.HOUGH_GRADIENT,dp=1,minDist=60,param1=60,param2=14,minRadius=9,maxRadius=18)
        out[s]=[] if c is None else [(float(x+x0),float(y),float(r)) for x,y,r in c[0]]
    return out
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(work,list(enumerate(idx)),chunksize=8)
    json.dump(R,open('holes_raw.json','w'))
    for s in 'LR':
        n=np.array([len(r[s]) if r[s] is not None else -1 for r in R])
        print(s,'skipped (coil)',(n<0).sum(),'count hist',np.bincount(n[n>=0],minlength=8)[:8])
