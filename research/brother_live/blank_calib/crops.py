import json,sys,numpy as np,cv2,pikepdf,pickle
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/brother_live/work'; _os.makedirs(_WORK, exist_ok=True)
sys.path.insert(0,_ROOT + '/src')
from scanpp import geometry, pipeline, pdfio
S=_WORK
R=_ROOT + '/research/r4_geometry/'
I=json.load(open(R+'index.json'));N=dict((i,r[0]) for i,r in json.load(open(S+'names.json')))
CFG={34:(260,-30,22),591:(150,-14,25),850:(250,-30,25),866:(150,-30,10),1040:(250,-30,8),1126:(150,-12,20),1288:(300,-30,25),1474:(300,-30,25),1705:(250,-30,25),1801:(250,-30,25),1857:(300,-30,25)}
out={}
for i,(xm,y0,y1) in CFG.items():
    e=I[i];pdf=pikepdf.open(_ROOT + '/'+e['pdf'])
    t=next(pdfio.iter_page_inputs(pdf,[e['page']]));rgb=pipeline._decode(t);s=t.dpi/100
    l,tp,w,h=N[i]
    x0=int((l+w+3)*s);x1=int((l+w+xm)*s);ya=int((tp+y0)*s);yb=int((tp+h+y1)*s)
    c=rgb[ya:yb,x0:x1];g=cv2.cvtColor(c,cv2.COLOR_RGB2GRAY)
    bg=cv2.medianBlur(cv2.dilate(g,np.ones((25,25),np.uint8)),5).astype(np.float32)
    ink=np.clip(1-g/np.maximum(bg,1),0,1)
    ink[ink<0.06]=0
    m=(ink>0.12).astype(np.uint8)*255
    ln=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((1,int(0.6*t.dpi)),np.uint8))
    ln=cv2.dilate(ln,np.ones((5,3),np.uint8))
    ink[ln>0]=0
    # drop tiny specks
    n,lab,st,_=cv2.connectedComponentsWithStats((ink>0.12).astype(np.uint8))
    keep=np.zeros(n,bool)
    for k in range(1,n):
        if st[k,4]>=25: keep[k]=True
    ink[~keep[lab]]=0
    ys,xs=np.where(ink>0.12);
    ink=ink[ys.min():ys.max()+1,xs.min():xs.max()+1]
    p=np.percentile(ink[ink>0.12],95);a=np.clip(ink/p,0,1).astype(np.float32)
    out[i]=(a,t.dpi);print(i,a.shape,t.dpi)
pickle.dump(out,open(S+'crops.pkl','wb'))
sh=[cv2.copyMakeBorder((255*(1-a)).astype(np.uint8),5,5,5,5,0,value=255) for a,_ in out.values()]
W=max(x.shape[1] for x in sh);sh=[cv2.copyMakeBorder(x,0,0,0,W-x.shape[1],0,value=255) for x in sh]
cv2.imwrite(S+'crops.png',np.vstack(sh))
