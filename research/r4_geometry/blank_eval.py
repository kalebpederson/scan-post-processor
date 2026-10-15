import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
def w(m):
    g=cv2.imread(m['fn'],0);t=time.perf_counter();f=G.blank_features(g,100);dt=(time.perf_counter()-t)*1000
    return f[:6],dt
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('blank_feats2.json','w'))
    lab=json.load(open('labels_blank.json'))['labels']
    blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
    s=np.array([r[0][0]['mass'] if r[0] else 0 for r in R])
    print('BLANK  max %.1f p99 %.1f p90 %.1f median %.2f'%(s[blank].max(),np.percentile(s[blank],99),np.percentile(s[blank],90),np.median(s[blank])))
    print('NONBLANK min %.1f p1 %.1f p5 %.1f'%(s[~blank].min(),np.percentile(s[~blank],1),np.percentile(s[~blank],5)))
    o=np.argsort(s);print('top blanks',[(int(i),round(s[i],1)) for i in o if blank[i]][-6:]);print('lowest nonblank',[(int(i),round(s[i],1)) for i in o if not blank[i]][:8])
    print('median ms',np.median([r[1] for r in R]))
