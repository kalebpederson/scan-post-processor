import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
from methods import METHODS

def w(m):
    g=cv2.imread(m['fn'],0); r={}
    for n,f in METHODS.items():
        t=time.perf_counter(); a,c=f(g,100); r[n]=(float(a),float(c),(time.perf_counter()-t)*1000)
    return r
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(16) as p: res=p.map(w,idx,chunksize=8)
    json.dump(res,open('survey_raw3.json','w'))
    for n in METHODS: print(n,np.mean([r[n][2] for r in res]))
