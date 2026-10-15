import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
def per(run):
    x=run.astype(float)-np.convolve(run,np.ones(81)/81,mode='same')   # detrend (kills wedge ramps)
    x=x[40:-40]
    if len(x)<100 or x.std()<1e-6: return 0.0,0.0
    F=np.abs(np.fft.rfft(x*np.hanning(len(x))))**2;fr=np.fft.rfftfreq(len(x))
    band=(fr>1/45)&(fr<1/9)
    return float(F[band].max()/F[1:].sum()),float(1/fr[band][F[band].argmax()])
def w(m):
    g=cv2.imread(m['fn'],0);pl=G._paper_level(g);r={}
    for s in 'LR':
        run=G._edge_runs(g,s,0.75*pl,60)
        p,per_=per(run);r[s]=(float(np.mean(run>=10)),float(np.mean(run<=2)),p,per_,float(run.mean()))
    return r
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('coil2.json','w'))
    coil=json.load(open('coil_raw.json'))
    for s in 'LR':
        c0=np.array([coil[i][s][0]>=0.3 for i in range(len(idx))]);f10=np.array([r[s][0] for r in R]);pp=np.array([r[s][2] for r in R])
        print(s,'old coil',c0.sum(),'| periodicity>0.25',(pp>0.25).sum(),'| both',((pp>0.25)&c0).sum(),'| old only',(c0&(pp<=0.25)).sum(),'| new only',((pp>0.25)&~c0).sum())
        print('  period (px) of coil pages median',np.median([r[s][3] for r,c in zip(R,c0) if c]),'periodicity pct among old coil',np.round(np.percentile(pp[c0],[5,25,50]),2),' among others pcts 90/99/max',np.round(np.percentile(pp[~c0],[90,99,100]),2))
