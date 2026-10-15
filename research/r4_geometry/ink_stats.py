import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from methods import ink_map
def w(m):
    g=cv2.imread(m['fn'],0); h,wd=g.shape
    ink=ink_map(g,100)
    mb=int(100*0.5)
    core=ink[mb:h-mb,mb:wd-mb]
    r={}
    for T in (0.06,0.1,0.15,0.25):
        r[f'f{T}']=float((core>T).mean())
        r[f'ffull{T}']=float((ink>T).mean())
    b=(core>0.12).astype(np.uint8)
    n,lab,st,_=cv2.connectedComponentsWithStats(b,connectivity=8)
    a=st[1:,4] if n>1 else np.array([0])
    r['maxcc']=int(a.max()); r['ncc']=int((a>=4).sum()); r['ncc20']=int((a>=20).sum())
    # sharpness: gradient magnitude 99.9 pct on core
    gb=cv2.GaussianBlur(g,(0,0),0.7).astype(np.float32)
    gm=np.hypot(cv2.Sobel(gb,cv2.CV_32F,1,0),cv2.Sobel(gb,cv2.CV_32F,0,1))[mb:h-mb,mb:wd-mb]
    r['grad999']=float(np.percentile(gm,99.9)); r['std']=float(core.std()); r['med']=float(np.median(g))
    return r
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(16) as p: res=p.map(w,idx,chunksize=8)
    json.dump(res,open('ink_raw.json','w'))
