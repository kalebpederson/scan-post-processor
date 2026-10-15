import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from methods import ink_map
cv2.setNumThreads(1)
def comps(g,dpi=100,T=0.06,margin=0.0):
    ink=ink_map(g,dpi); ink=cv2.GaussianBlur(ink,(0,0),0.6)
    h,w=ink.shape
    b=(ink>T).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(b,connectivity=8)
    gb=cv2.GaussianBlur(g,(0,0),0.7).astype(np.float32)
    gm=np.hypot(cv2.Sobel(gb,cv2.CV_32F,1,0),cv2.Sobel(gb,cv2.CV_32F,0,1))
    out=[]
    for i in range(1,n):
        x,y,ww,hh,ar=st[i]
        if ar<4: continue
        m=lab[y:y+hh,x:x+ww]==i
        sub=ink[y:y+hh,x:x+ww][m]
        out.append(dict(x=int(x),y=int(y),w=int(ww),h=int(hh),area=int(ar),peak=float(sub.max()),mean=float(sub.mean()),mass=float(sub.sum()),
            grad=float(gm[y:y+hh,x:x+ww][m].max())))
    return out,ink
def w(m):
    g=cv2.imread(m['fn'],0)
    c,_=comps(g)
    c.sort(key=lambda d:-d['mass'])
    return c[:40], len(c)
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(12) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('comp_raw.json','w'))
