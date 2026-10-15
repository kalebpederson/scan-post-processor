"""Method (d): paper-edge (feeder background wedge) skew estimate from the dark strip profile."""
import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
SIGN={'L':+1,'R':-1,'T':-1,'B':+1}
def est_edge(g,d=100):
    pl=G._paper_level(g);ests=[];ws=[]
    for s in 'LRTB':
        run=G._edge_runs(g,s,0.75*pl,60).astype(float)
        n=len(run);t=np.arange(n)
        k=(run>=2)&(run<=0.35*d)
        if k.sum()<0.3*n or run[k].max()<0.05*d: continue         # need a long wedge, >=1.3mm deep
        # robust linear fit (Theil-Sen-like via RANSAC-lite: iterate trimming)
        tt,rr=t[k],run[k]
        p=np.polyfit(tt,rr,1)
        for _ in range(2):
            res=np.abs(rr-np.polyval(p,tt));m=res<=max(1.5,np.percentile(res,75))
            if m.sum()<20: break
            p=np.polyfit(tt[m],rr[m],1)
        if np.polyval(p,tt).max()-np.polyval(p,tt).min()<2: continue   # flat strip: no slope information
        ests.append(SIGN[s]*np.degrees(np.arctan(p[0])));ws.append(k.sum())
    if not ests: return None
    return float(np.average(ests,weights=ws)),len(ests)
def w(m):
    return est_edge(cv2.imread(m['fn'],0))
if __name__=='__main__':
    idx=json.load(open('index.json'));F=json.load(open('final_survey.json'))
    lab=json.load(open('labels_blank.json'))['labels']
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('edge_skew.json','w'))
    ang=np.array([r['ang'] for r in F]);conf=np.array([r['conf'] for r in F])
    blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
    ok=np.array([r is not None for r in R]);nb=~blank
    print('edge method applicable on',ok.sum(),'pages (%.1f%%)'%(ok.mean()*100),' nonblank',(ok&nb).sum())
    k=ok&nb&(conf>=0.6)
    e=np.array([R[i][0] for i in np.where(k)[0]]);a=ang[k];d=np.abs(e-a)
    print('vs consensus (reliable n=%d): MAE %.3f median %.3f p90 %.3f within 0.3: %.1f%%, corr %.2f'%(k.sum(),d.mean(),np.median(d),np.percentile(d,90),(d<0.3).mean()*100,np.corrcoef(e,a)[0,1]))
    big=k.copy();big[k]=np.abs(a)>0.5
    e2=np.array([R[i][0] for i in np.where(big)[0]]);a2=ang[big];print('pages |consensus|>0.5: n=%d edge applicable-> MAE %.3f within 0.3: %.1f%%'%(big.sum(),np.abs(e2-a2).mean(),(np.abs(e2-a2)<0.3).mean()*100))
    allbig=nb&(conf>=0.6)&(np.abs(ang)>0.5);print('of',allbig.sum(),'pages with |a|>0.5, edge method applicable on %.1f%%'%(ok[allbig].mean()*100))
