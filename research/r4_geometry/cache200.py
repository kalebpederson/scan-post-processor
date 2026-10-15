import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from fullres import get_full
cv2.setNumThreads(1)
idx=json.load(open('index.json'));lab=json.load(open('labels_blank.json'))['labels']
def w(i):
    m=idx[i];im=get_full(m);dpi=m['dpi']
    if dpi>210:
        s=200/dpi;im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'cache200/{i}.png',im[:,:,::-1]);return i,max(dpi,0)
if __name__=='__main__':
    ids=[int(k) for k in lab]
    with Pool(10) as p: R=p.map(w,ids,chunksize=4)
    json.dump({'ids':ids,'dpi':{str(i):d for i,d in R}},open('cache200.json','w'))
    print(len(ids))
