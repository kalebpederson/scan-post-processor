import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from edges import strip_profile,paper_level
cv2.setNumThreads(1)
def longest(mask):
    best=0;cur=0
    for v in mask:
        cur=cur+1 if v else 0
        best=max(best,cur)
    return best
def work(m):
    g=cv2.imread(m['fn'],0);pl=paper_level(g);r={}
    for s in 'LRTB':
        run=strip_profile(g,s,pl,frac=0.6,maxd=80)
        for w in (2,4,8,16):
            r[f'seg{w}_{s}']=longest(run>=w)
        r['runmax_'+s]=int(run.max())
        r['runp90_'+s]=float(np.percentile(run,90))
    return r
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(work,idx,chunksize=8)
    json.dump(R,open('edges2_raw.json','w'))
