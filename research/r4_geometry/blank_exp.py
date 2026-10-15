import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,itertools
from multiprocessing import Pool
import geometry as G
from blank_synth import draw,TYPES,DELTAS
cv2.setNumThreads(1)
idx=json.load(open('index.json'));lab=json.load(open('labels_blank.json'))['labels']
def eff_dpi(i): return 200.0 if idx[i]['dpi']>210 else float(idx[i]['dpi'])
CONF=[(100,0.12,60),(100,0.12,30),(150,0.12,45),(150,0.12,30),(200,0.12,45),(200,0.12,30),(200,0.12,20),(200,0.08,30)]
def feats(g,d,wd,T,gt):
    """returns total sharp mass, max cluster sharp mass, max cluster diag (in)"""
    if d>wd*1.05:
        s=wd/d;g=cv2.resize(g,None,fx=s,fy=s,interpolation=cv2.INTER_AREA);d=wd
    emask,_=G.edge_artifact_mask(g,d)
    ink=cv2.GaussianBlur(G.ink_map(g,d),(0,0),0.6*d/100)
    gb=cv2.GaussianBlur(g,(0,0),0.7*d/100).astype(np.float32)
    gm=np.hypot(cv2.Sobel(gb,cv2.CV_32F,1,0),cv2.Sobel(gb,cv2.CV_32F,0,1))
    ok=(ink>T)&(gm>gt)&~(cv2.dilate(emask.astype(np.uint8),np.ones((3,3),np.uint8))>0)
    m=np.where(ok,ink,0).astype(np.float32);sc=(100.0/d)**2
    tot=float(m.sum()*sc)
    k=max(3,int(round(0.07*d)))|1
    dil=cv2.dilate(ok.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    n,lab_,st,_=cv2.connectedComponentsWithStats(dil,connectivity=8)
    best=0;bd=0;ext=0
    for i in range(1,n):
        x,y,w,h,_a=st[i];mm=m[y:y+h,x:x+w][lab_[y:y+h,x:x+w]==i].sum()*sc
        diag=np.hypot(w,h)/d
        best=max(best,mm);bd=max(bd,diag)
        # mass of clusters at least 0.2in across
        if diag>=0.2: ext=max(ext,mm)
    return tot,float(best),float(bd),float(ext)
def work(i):
    g=cv2.imread(f'cache200/{i}.png');d=eff_dpi(i)
    rgb=g[:,:,::-1].copy();gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    out={'i':i,'cl':lab[str(i)]['label'],'base':{str(c):feats(gray,d,*c) for c in CONF}}
    if out['cl']=='BLANK' and i in SYN:
        h,w=gray.shape;rs=np.random.RandomState(i);paper=np.percentile(rgb.reshape(-1,3),90,axis=0)
        syn={}
        for k in TYPES:
            if k=='pagenum': x=int(rs.uniform(0.4,0.6)*w);y=int(h-0.3*d)
            else: x=int(rs.uniform(0.15,0.7)*w);y=int(rs.uniform(0.15,0.8)*h)
            layer=np.zeros((h,w),np.uint8);draw(layer,k,x,y,d,1.0);layer=cv2.GaussianBlur(layer.astype(np.float32)/255,(0,0),0.6*d/200)
            for dl in DELTAS:
                t=np.clip(rgb.astype(np.float32)-layer[:,:,None]*dl*(paper/255.0)[None,None,:],0,255).astype(np.uint8)
                tg=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
                syn[f'{k}_{dl}']={str(c):feats(tg,d,*c) for c in CONF}
        out['syn']=syn
    return out
bl_=[int(k) for k,v in lab.items() if v['label']=='BLANK']
SYN=set(int(x) for x in np.random.RandomState(5).choice(bl_,24,replace=False))
if __name__=='__main__':
    ids=[int(k) for k in lab]
    with Pool(10) as p: R=p.map(work,ids,chunksize=2)
    json.dump(R,open('blank_exp.json','w'))
