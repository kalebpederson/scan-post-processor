import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,itertools
from multiprocessing import Pool
from methods import _peak_hist
cv2.setNumThreads(1)
idx=json.load(open('index.json'))
def segs(m):
    g=cv2.imread(m['fn'],0);h,w=g.shape;g=g[25:h-25,50:w-50]
    d=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);L=d.detect(g)[0]
    L=L.reshape(-1,4) if L is not None else np.zeros((0,4))
    ln=np.hypot(L[:,2]-L[:,0],L[:,3]-L[:,1]);return L[ln>=15].astype(np.float32)
if __name__=='__main__':
    with Pool(10) as p: S=p.map(segs,idx,chunksize=8)
    import pickle;pickle.dump(S,open('lsd_segs.pkl','wb'))
