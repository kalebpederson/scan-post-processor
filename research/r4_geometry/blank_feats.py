import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
import methods
from methods import ink_map
from coil import band_stats
from edges import paper_level
cv2.setNumThreads(1)
def feats(g,dpi=100,T=0.12):
    pl=paper_level(g)
    h,w=g.shape
    ink=cv2.GaussianBlur(ink_map(g,dpi),(0,0),0.6)
    # coil / binding mask
    mask=np.ones_like(ink,bool)
    for s in 'LR':
        cov,depth=band_stats(g,pl,s)
        if cov>=0.3 and depth>=15:
            d=depth+10
            if s=='L': mask[:,:d]=False
            else: mask[:,w-d:]=False
    b=((ink>T)&mask).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(b,connectivity=8)
    keep=np.zeros(n,bool)
    for i in range(1,n):
        x,y,ww,hh,ar=st[i]
        tlr=(x<=1 or x+ww>=w-1); ttb=(y<=1 or y+hh>=h-1)
        if (tlr or ttb) and min(ww,hh)<=40: continue    # thin strip hugging a page edge
        keep[i]=True
    b2=keep[lab].astype(np.uint8)
    # merge strokes into clusters
    dil=cv2.dilate(b2,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    n2,lab2,st2,_=cv2.connectedComponentsWithStats(dil,connectivity=8)
    gb=cv2.GaussianBlur(g,(0,0),0.7).astype(np.float32)
    gm=np.hypot(cv2.Sobel(gb,cv2.CV_32F,1,0),cv2.Sobel(gb,cv2.CV_32F,0,1))
    out=[]
    for i in range(1,n2):
        x,y,ww,hh,ar=st2[i]
        sub=(lab2[y:y+hh,x:x+ww]==i)&(b2[y:y+hh,x:x+ww]>0)
        if sub.sum()<4: continue
        v=ink[y:y+hh,x:x+ww][sub]
        out.append(dict(x=int(x),y=int(y),w=int(ww),h=int(hh),npx=int(sub.sum()),mass=float(v.sum()),peak=float(v.max()),
                        p90=float(np.percentile(v,90)),grad=float(np.percentile(gm[y:y+hh,x:x+ww][sub],90))))
    out.sort(key=lambda c:-c['mass'])
    return out[:12]
def work(m):
    g=cv2.imread(m['fn'],0); return feats(g,100)
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(8) as p: R=p.map(work,idx,chunksize=8)
    json.dump(R,open('blank_feats.json','w'))
