import json, sys, numpy as np, cv2
from multiprocessing import Pool
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/brother_live/work'; _os.makedirs(_WORK, exist_ok=True)
sys.path.insert(0,_ROOT + '/src')
from scanpp import geometry
R=_ROOT + '/research/r4_geometry/'
idx=json.load(open(R+'index.json'))
def f(i):
    im=cv2.imread(R+idx[i]['fn'],cv2.IMREAD_COLOR)
    if im is None: return i,None,None
    rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    raw=geometry.blank_score(rgb,100.0)
    out,info,g,d=geometry._clean_edges_g(rgb,100.0,inplace=False)
    pipe=geometry.blank_score(g if g is not None else out,d if g is not None else 100.0)
    return i,raw,pipe
if __name__=='__main__':
    with Pool(12) as p: res=p.map(f,range(len(idx)),chunksize=8)
    json.dump(res,open(_WORK + '/scores.json','w'))
