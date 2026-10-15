"""Synthetic faint pencil marks on real blank pages -> sensitivity of the blank detector."""
import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
from fullres import get_full
import geometry as G
cv2.setNumThreads(1)
idx=json.load(open('index.json'))
lab=json.load(open('labels_blank.json'))['labels']
TYPES=['pagenum','word','tick','dash','cross','dot']
DELTAS=[25,40,60,90,140]
def draw(layer,kind,x,y,dpi,val):
    """draw into 'layer' (float 0..1 mask) single-stroke pencil-like marks"""
    th=max(1,int(round(dpi*0.010)))   # ~0.25 mm pencil line
    S=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    if kind=='pagenum': cv2.putText(layer,'14',(x,y),cv2.FONT_HERSHEY_SIMPLEX,dpi*0.14/22,255,th,cv2.LINE_AA)
    elif kind=='word': cv2.putText(layer,'Check',(x,y),S,dpi*0.22/22,255,th,cv2.LINE_AA)
    elif kind=='tick':
        s=int(dpi*0.25);pts=np.array([[x,y],[x+s//3,y+s//2],[x+s,y-s//2]],np.int32);cv2.polylines(layer,[pts],False,255,th,cv2.LINE_AA)
    elif kind=='dash': cv2.line(layer,(x,y),(x+int(dpi*0.4),y-int(dpi*0.03)),255,th,cv2.LINE_AA)
    elif kind=='cross':
        s=int(dpi*0.3);cv2.line(layer,(x,y),(x+s,y+s),255,th,cv2.LINE_AA);cv2.line(layer,(x+s,y),(x,y+s),255,th,cv2.LINE_AA)
    elif kind=='dot': cv2.circle(layer,(x,y),int(dpi*0.03),255,-1,cv2.LINE_AA)
def work(i):
    m=idx[i];im=get_full(m);dpi=m['dpi'];h,w=im.shape[:2]
    rs=np.random.RandomState(i)
    paper=np.percentile(im.reshape(-1,3),90,axis=0)
    base=G.blank_score(im,dpi)
    out={'i':i,'base':base,'dpi':dpi,'res':{}}
    for k in TYPES:
        # position: random interior, or bottom-margin for pagenum
        if k=='pagenum': x=int(rs.uniform(0.4,0.6)*w);y=int(h-0.3*dpi)
        else: x=int(rs.uniform(0.15,0.7)*w);y=int(rs.uniform(0.15,0.8)*h)
        layer=np.zeros((h,w),np.uint8);draw(layer,k,x,y,dpi,1.0);layer=layer.astype(np.float32)/255
        layer=cv2.GaussianBlur(layer,(0,0),0.6*dpi/200)
        layer=np.clip(layer*1.0,0,1)
        for d in DELTAS:
            a=layer[:,:,None]*d
            test=np.clip(im.astype(np.float32)-a*(paper/255.0)[None,None,:],0,255).astype(np.uint8)
            out['res'][f'{k}_{d}']=G.blank_score(test,dpi)
    return out
if __name__=='__main__':
    bl=[int(k) for k,v in lab.items() if v['label']=='BLANK']
    rs=np.random.RandomState(5);sel=[int(x) for x in rs.choice(bl,48,replace=False)]
    with Pool(10) as p: R=p.map(work,sel,chunksize=2)
    json.dump(R,open('blank_synth.json','w'))
    thr=G.BLANK_THRESHOLD
    print('thr',thr,'base scores: max %.1f'%max(r['base'] for r in R))
    print('detection rate (score>=thr) by type x delta (gray levels darker than paper); median score in brackets')
    print('type     '+'  '.join(f'd={d:3d}      ' for d in DELTAS))
    for k in TYPES:
        row=[]
        for d in DELTAS:
            v=np.array([r['res'][f'{k}_{d}'] for r in R]);row.append(f"{(v>=thr).mean()*100:5.1f}% [{np.median(v):5.0f}]")
        print(f'{k:8s} '+'  '.join(row))
