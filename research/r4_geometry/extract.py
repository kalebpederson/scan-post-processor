"""Extract every scan page to a reduced-resolution cache (target ~100 ppi).
Writes cache/NNNN.png (RGB, ~100ppi) and index.json."""
import json, sys, io, os
import numpy as np, pikepdf, cv2
from multiprocessing import Pool
from pikepdf import PdfImage
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r4_geometry/work'; _os.makedirs(_WORK, exist_ok=True)
ROOT=_ROOT
TARGET=100
def work(a):
    k,pdfpath=a
    out=[]
    try:
        pdf=pikepdf.open(os.path.join(ROOT,pdfpath))
    except Exception as e:
        return [(k,0,None,str(e))]
    for pi,page in enumerate(pdf.pages):
        try:
            imgs=list(page.images.items())
            if not imgs: out.append((k,pi,None,'noimg')); continue
            name,raw=max(imgs,key=lambda kv:int(kv[1].Width)*int(kv[1].Height))
            pim=PdfImage(raw).as_pil_image().convert('RGB')
            a_=np.asarray(pim)
            w,h=pim.size
            mb=[float(x) for x in page.MediaBox]
            wpt=mb[2]-mb[0]
            dpi=w/(wpt/72)
            s=TARGET/dpi
            sm=cv2.resize(a_,(round(w*s),round(h*s)),interpolation=cv2.INTER_AREA)
            fn=f'cache/{k:03d}_{pi:02d}.png'
            cv2.imwrite(fn,sm[:,:,::-1])
            out.append((k,pi,dict(fn=fn,w=w,h=h,dpi=dpi,nimg=len(imgs),sw=sm.shape[1],sh=sm.shape[0]),None))
        except Exception as e:
            out.append((k,pi,None,repr(e)))
    return out
if __name__=='__main__':
    d=json.load(open(f'{ROOT}/research/survey.json'))
    sc=[x for x in d['pdfs'] if x['class']=='scan']
    with Pool(16) as p:
        res=p.map(work,[(i,x['path']) for i,x in enumerate(sc)],chunksize=2)
    idx=[]
    err=0
    for r in res:
        for k,pi,m,e in r:
            if m is None: err+=1; print('ERR',sc[k]['path'],pi,e); continue
            m.update(pdf=sc[k]['path'],page=pi,npages=sc[k]['pages'],k=k); idx.append(m)
    json.dump(idx,open('index.json','w'))
    print(len(idx),'pages',err,'errors')
