import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
idx=json.load(open('index.json'));R0=json.load(open('gt_results.json'))
trials=[(r['i'],r['a']) for r in R0]
def w(t):
    i,a=t
    im=cv2.imread(idx[i]['fn']);h,wd=im.shape[:2]
    r=cv2.warpAffine(im,cv2.getRotationMatrix2D((wd/2,h/2),a,1.0),(wd,h),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    e=G.skew_estimates(g,100);ang,conf=G.estimate_skew(g,100)
    return dict(i=i,a=a,est=ang,conf=conf,e=e)
def wb(i):
    g=cv2.imread(idx[i]['fn'],0);e=G.skew_estimates(g,100);ang,conf=G.estimate_skew(g,100);return i,dict(est=ang,conf=conf,e=e)
if __name__=='__main__':
    with Pool(8) as p:
        R=p.map(w,trials,chunksize=2)
        B=dict(p.map(wb,sorted(set(i for i,_ in trials))))
    json.dump(dict(R=R,B=B),open('gt_final.json','w'))
    a=np.array([r['a'] for r in R]);est=np.array([r['est'] for r in R]);conf=np.array([r['conf'] for r in R])
    base=np.array([B[r['i']]['est'] for r in R])
    for name,m in (('|a|<=5 (n=%d)'%(np.abs(a)<=5).sum(),np.abs(a)<=5),('|a|<=2.5',np.abs(a)<=2.5),('2.5<|a|<=5',(np.abs(a)>2.5)&(np.abs(a)<=5)),('5<|a|<=8',np.abs(a)>5)):
        for nm2,acc in (('all',np.ones(len(a),bool)),('conf>=0.6',conf>=0.6)):
            k=m&acc
            if k.sum()==0: continue
            e=np.abs(est-a)[k];rel=np.abs(est-base-a)[k]
            print(f"{name:18s} {nm2:9s} n={k.sum():3d} abs: mean {e.mean():.3f} p95 {np.percentile(e,95):.3f} max {e.max():.2f} fail>0.3 {(e>0.3).mean()*100:.1f}% | self-consistency mean {rel.mean():.3f} p95 {np.percentile(rel,95):.3f} max {rel.max():.2f}")
    k=np.abs(a)<=5;print('accepted(conf>=.6) among |a|<=5: %.1f%%'%((conf[k]>=0.6).mean()*100),' among |a|>5 flagged unreliable or >max: %.1f%%'%(((conf[~k]<0.6)|(np.abs(est[~k])>5)).mean()*100))
    # which individual est is each: error stats
    for n in ('proj','lsd','cc','grad'):
        ee=np.array([abs(r['e'][n]-r['a']) for r in R])[np.abs(a)<=5];print(n,'|a|<=5 mean %.3f p95 %.3f fail>0.3 %.1f%%'%(ee.mean(),np.percentile(ee,95),(ee>0.3).mean()*100))
