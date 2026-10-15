import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
import geometry as G
cv2.setNumThreads(1)
def w(m):
    g=cv2.imread(m['fn'],0)
    mask,info=G.edge_artifact_mask(g,100)
    # content damage proxy: painted pixels that are "ink" and NOT in the outer 0.12in band nor coil
    return dict(frac=float(mask.mean()),info=info)
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(w,idx,chunksize=8)
    json.dump(R,open('edge_eval.json','w'))
    fr=np.array([r['frac'] for r in R])
    print('painted frac pcts 50/90/99/max',np.round(np.percentile(fr,[50,90,99,100])*100,2),'%')
    coil=np.array([('coil_L' in r['info']) or ('coil_R' in r['info']) for r in R])
    cl=np.array(['coil_L' in r['info'] for r in R]);cr=np.array(['coil_R' in r['info'] for r in R])
    print('coil pages',coil.sum(),'L',cl.sum(),'R',cr.sum(),'both',(cl&cr).sum(),'depth in pct (L)',np.round(np.percentile([r['info']['coil_L'] for r in R if 'coil_L' in r['info']],[5,50,95]),2))
    print('non-coil pages painted frac pcts 50/90/99/max',np.round(np.percentile(fr[~coil],[50,90,99,100])*100,3),'%')
    for s in 'LRTB':
        st=np.array([r['info']['strip_'+s] for r in R])
        print(s,'strip depth (99th pct along edge, in) >0.02in %.3f >0.04 %.3f >0.08(2mm) %.3f >0.16(4mm) %.3f'%tuple((st>t).mean() for t in (0.02,0.04,0.08,0.16)))
