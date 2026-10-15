import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
def feats(g,T=0.12):
    d=100.0;h,w=g.shape
    emask,_=G.edge_artifact_mask(g,d)
    ink=cv2.GaussianBlur(G.ink_map(g,d),(0,0),0.6)
    ink[cv2.dilate(emask.astype(np.uint8),np.ones((3,3),np.uint8))>0]=0
    gb=cv2.GaussianBlur(g,(0,0),0.7).astype(np.float32)
    gm=np.hypot(cv2.Sobel(gb,cv2.CV_32F,1,0),cv2.Sobel(gb,cv2.CV_32F,0,1))
    b=(ink>T).astype(np.uint8)
    dil=cv2.dilate(b,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n,lab,st,_=cv2.connectedComponentsWithStats(dil,connectivity=8)
    out=[]
    for i in range(1,n):
        x,y,ww,hh,_a=st[i]
        sub=(lab[y:y+hh,x:x+ww]==i)&(b[y:y+hh,x:x+ww]>0)
        if sub.sum()<4: continue
        v=ink[y:y+hh,x:x+ww][sub];gg=gm[y:y+hh,x:x+ww][sub]
        r=dict(x=int(x),y=int(y),w=int(ww),h=int(hh),mass=float(v.sum()),peak=float(v.max()),npx=int(sub.sum()),g90=float(np.percentile(gg,90)),gmax=float(gg.max()))
        for g0 in (20,30,45,60): r[f'ms{g0}']=float(v[gg>g0].sum())
        out.append(r)
    out.sort(key=lambda c:-c['mass']);return out
def w(m): return feats(cv2.imread(m['fn'],0))
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('blank_feats3.json','w'))
