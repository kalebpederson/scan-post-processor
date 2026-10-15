import json
from multiprocessing import Pool
import corpus_eval as C
from lib import *
EXT=[50,60,75,90]
def job(f):
    g=cv2.imread(f'{CORPUS}/{f}',0); ref=C.words(g,300,90); rows=[]
    for r in EXT:
        up=upsample(downsample(g,300,r),(g.shape[1],g.shape[0])); rows.append(dict(r=r,rt=C.recall(ref,C.words(up,300))))
    return dict(file=f,nref=len(ref),rows=rows)
if __name__=='__main__':
    with Pool(14) as p: out=p.map(job,C.FILES,chunksize=1)
    json.dump(out,open(f'{HERE}/corpus_ext_results.json','w'),indent=1)
