import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,itertools,time
from multiprocessing import Pool
from methods import _peak_hist
cv2.setNumThreads(1)
idx=json.load(open('index.json'));cand=json.load(open('gt_candidates.json'))
R=json.load(open('gt_results.json'))
def segs(t):
    i,a=t
    im=cv2.imread(idx[i]['fn']);h,w=im.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
    r=cv2.warpAffine(im,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=(255,255,255));g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    out={}
    for nm,ref in (('std',cv2.LSD_REFINE_STD),('adv',cv2.LSD_REFINE_ADV)):
        d=cv2.createLineSegmentDetector(ref);t0=time.perf_counter();L=d.detect(g)[0];dt=time.perf_counter()-t0
        out[nm]=(L.reshape(-1,4) if L is not None else np.zeros((0,4)),dt)
    # also upscaled-by-1.5 not now
    return out
if __name__=='__main__':
    with Pool(12) as p: S=p.map(segs,[(r['i'],r['a']) for r in R],chunksize=4)
    A=np.array([r['a'] for r in R])
    def est(L,minlen,wp,sm,mode,rng=12.0,bin_=0.1):
        dx=L[:,2]-L[:,0];dy=L[:,3]-L[:,1];ln=np.hypot(dx,dy);k=ln>=100*minlen
        dx,dy,ln=dx[k],dy[k],ln[k]
        ang=np.degrees(np.arctan2(-dy,dx));ang=(ang+90)%180-90
        dev=np.where(np.abs(ang)<=45,ang,ang-np.sign(ang)*90)
        if mode=='h': m=np.abs(ang)<=rng
        else: m=np.abs(dev)<=rng; 
        d=(ang if mode=='h' else dev)
        return _peak_hist(d[m],ln[m]**wp,-rng,rng,bin_,sm)[0]
    res=[]
    for ref in ('std','adv'):
        for minlen,wp,sm,mode in itertools.product((0.15,0.3,0.5,0.8),(1,2,3),(0.2,0.35,0.5),('hv','h')):
            e=np.abs(np.array([est(s[ref][0],minlen,wp,sm,mode) for s in S])-A)
            res.append((e.mean(),np.percentile(e,95),e.max(),(e>0.3).mean(),ref,minlen,wp,sm,mode))
    res.sort()
    for r in res[:12]: print(np.round(r[:4],3),r[4:])
    print('worst',[np.round(r[:4],3) for r in res[-2:]])
    print('time std/adv ms',np.median([s['std'][1] for s in S])*1000,np.median([s['adv'][1] for s in S])*1000)
