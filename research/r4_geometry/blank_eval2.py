import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
def w(m):
    g=cv2.imread(m['fn'],0);t=time.perf_counter();s=G.blank_score(g,100);return s,(time.perf_counter()-t)*1000
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    s=np.array([r[0] for r in R]);json.dump(s.tolist(),open('blank_scores.json','w'))
    lab=json.load(open('labels_blank.json'))['labels']
    blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
    print('BLANK  max %.1f p99 %.1f median %.2f | NB min %.1f p1 %.1f'%(s[blank].max(),np.percentile(s[blank],99),np.median(s[blank]),s[~blank].min(),np.percentile(s[~blank],1)))
    o=np.argsort(s);print('lowest NB',[(int(i),round(s[i],1)) for i in o if not blank[i]][:6]);print('highest blank',[(int(i),round(s[i],1)) for i in o if blank[i]][-5:])
    print('median ms',np.median([r[1] for r in R]))
