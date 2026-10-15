import json,subprocess,cv2,numpy as np,csv,io,sys
from multiprocessing import Pool
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/brother_live/work'; _os.makedirs(_WORK, exist_ok=True)
R=_ROOT + '/research/r4_geometry/'
I=json.load(open(R+'index.json'))
def f(i):
    im=cv2.imread(R+I[i]['fn']); top=im[:int(im.shape[0]*0.3)]
    top=cv2.resize(top,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    p=_WORK + '/tt/%d.png'%i
    cv2.imwrite(p,top)
    o=subprocess.run(['tesseract',p,'-','--psm','11','tsv'],capture_output=True,text=True,env={'OMP_THREAD_LIMIT':'1','PATH':'/usr/bin:/bin'}).stdout
    res=[]
    for r in csv.DictReader(io.StringIO(o),delimiter='\t',quoting=csv.QUOTE_NONE):
        if r['text'] and r['text'].strip().lower().rstrip(':_') in('name','name:'):
            res.append([int(r[k])/3 for k in('left','top','width','height')])
    return i,res
if __name__=='__main__':
    import os;os.makedirs(_WORK + '/tt',exist_ok=True)
    ids=[i for i,e in enumerate(I) if e['page']==0]
    with Pool(12) as p: out=p.map(f,ids)
    json.dump([o for o in out if o[1]],open(_WORK + '/names.json','w'))
    print(len(ids),sum(1 for o in out if o[1]))
