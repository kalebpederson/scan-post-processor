import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
def w(m):
    g=cv2.imread(m['fn'],0);t=time.process_time()
    est=G.skew_estimates(g,100);dt=time.process_time()-t
    names=G._VOTE_ORDER;e=np.array([est[k] for k in names])
    best=None
    for j in range(4):
        s=np.abs(e-e[j])<G.VOTE_TOL
        if best is None or s.sum()>best.sum(): best=s
    ww=G._VOTE_W*best;ang=float((e*ww).sum()/ww.sum());conf=(int(best.sum())-1)/3
    if abs(ang)>G.MAX_ANGLE: conf=0.0
    return dict(est=est,ang=ang,conf=conf,cpu_s=dt)
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('final_survey.json','w'))
