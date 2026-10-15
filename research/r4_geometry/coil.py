import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from edges import paper_level
cv2.setNumThreads(1)
def band_stats(g,pl,side,wpx=50):
    a={'L':g[:,:wpx],'R':g[:,-wpx:][:,::-1],'T':g[:wpx,:].T,'B':g[-wpx:,:][::-1,:].T}[side]
    dark=(a<0.45*pl)
    rowfrac=dark.mean(1)                   # along-edge position
    cov=float((rowfrac>0.2).mean())
    # extent: column profile of darkness over the length
    col=dark.mean(0)                       # by depth
    depth=int(np.sum(col>0.25))
    return cov,depth
def work(m):
    g=cv2.imread(m['fn'],0);pl=paper_level(g);return {s:band_stats(g,pl,s) for s in 'LRTB'}
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(work,idx,chunksize=8)
    json.dump(R,open('coil_raw.json','w'))
