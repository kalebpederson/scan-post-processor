import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
idx=json.load(open('index.json'));D=json.load(open('gt_final.json'))
trials=[(r['i'],r['a']) for r in D['R'] if abs(r['a'])<=5][::2]
def w(t):
    i,a=t;im=cv2.imread(idx[i]['fn']);h,wd=im.shape[:2]
    r=cv2.warpAffine(im,cv2.getRotationMatrix2D((wd/2,h/2),a,1.0),(wd,h),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY);ang,conf=G.estimate_skew(g,100)
    if not G.should_deskew(ang,conf): return a,ang,conf,None,False
    c=G.deskew(r[:,:,::-1],ang,fill=(255,255,255));g2=cv2.cvtColor(c[:,:,::-1].copy(),cv2.COLOR_BGR2GRAY)
    a2,c2=G.estimate_skew(g2,100);return a,ang,conf,a2,True
if __name__=='__main__':
    with Pool(8) as p: R=p.map(w,trials)
    rot=[r for r in R if r[4]];print('trials',len(R),'rotated',len(rot),'not rotated (unreliable/<0.25)',len(R)-len(rot))
    res=np.array([abs(r[3]) for r in rot]);pre=np.array([abs(r[0]) for r in rot])
    print('residual |angle| after correction: mean %.3f p50 %.3f p95 %.3f max %.2f; before mean %.2f'%(res.mean(),np.median(res),np.percentile(res,95),res.max(),pre.mean()))
    print('residual < 0.25 deg: %.1f%%'%((res<0.25).mean()*100))
    # timing deskew
    for d in (100,200,300):
        im=np.random.randint(0,255,(int(11*d),int(8.5*d),3),np.uint8);t=time.process_time();G.deskew(im,1.0,(255,255,255));print('deskew %d ppi: %.0f ms'%(d,(time.process_time()-t)*1000))
    g=np.random.randint(0,255,(2200,1700),np.uint8);print('estimate_skew on 200ppi input (incl downscale) CPU:',end=' ')
    im=cv2.imread(idx[0]['fn'],0);im=cv2.resize(im,None,fx=2,fy=2);t=time.process_time();G.estimate_skew(im,200);print('%.0f ms'%((time.process_time()-t)*1000))
