import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
from methods import METHODS
cv2.setNumThreads(1)
idx=json.load(open('index.json')); cand=json.load(open('gt_candidates.json'))
rs=np.random.RandomState(2024)
trials=[(i,float(rs.uniform(-8,8))) for i in cand for _ in range(3)]+[(i,float(rs.uniform(-2.5,2.5))) for i in cand for _ in range(2)]
def w(t):
    i,a=t
    im=cv2.imread(idx[i]['fn']); h,wd=im.shape[:2]
    M=cv2.getRotationMatrix2D((wd/2,h/2),a,1.0)   # content skew = +a (CCW)
    r=cv2.warpAffine(im,M,(wd,h),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    out={}
    for n,f in METHODS.items():
        t0=time.perf_counter(); e,c=f(g,100); out[n]=(float(e),float(c),(time.perf_counter()-t0)*1000)
    return dict(i=i,a=a,res=out)
if __name__=='__main__':
    with Pool(12) as p: R=p.map(w,trials,chunksize=2)
    json.dump(R,open('gt_results.json','w'))
