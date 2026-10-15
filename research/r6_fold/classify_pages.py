import sys,json; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *; from fold import ink_density
from multiprocessing import Pool
import warnings; warnings.filterwarnings('ignore')
def work(i):
    rgb,r=corpus_page(i); g=rgb.astype(np.float32); gm=g.mean(2)
    d=ink_density(gm,100.0)
    ch=(g.max(2)-g.min(2))/255.0
    ink=d>0.15
    n=ink.sum()
    strong=(d>0.5).sum()
    colored=((ch>0.25)&(d>0.25)).sum()
    return i,dict(ink=int(n),frac=float(n/d.size),strong=int(strong),col=int(colored),p99=float(np.percentile(d,99.5)),H=d.shape[0],W=d.shape[1])
if __name__=='__main__':
    with Pool(14) as p: res=dict(p.map(work,range(len(index())),chunksize=16))
    json.dump(res,open('classify.json','w'))
