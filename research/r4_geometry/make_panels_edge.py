import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,random
import geometry as G
from fullres import get_full
cv2.setNumThreads(1)
idx=json.load(open('index.json'));E=json.load(open('edge_eval.json'));F=json.load(open('final_survey.json'))
lab=json.load(open('labels_blank.json'))['labels']
nb=np.array([lab.get(str(i),{}).get('label')!='BLANK' for i in range(len(idx))])
coilL=[i for i in range(len(idx)) if 'coil_L' in E[i]['info'] and nb[i]];coilR=[i for i in range(len(idx)) if 'coil_R' in E[i]['info'] and nb[i]]
nc=[i for i in range(len(idx)) if not(('coil_L' in E[i]['info']) or ('coil_R' in E[i]['info'])) and nb[i]]
def mxs(i): return max(E[i]['info']['strip_'+s] for s in 'LRTB')
wedge=sorted(nc,key=lambda i:-mxs(i))[:60]
random.seed(8)
sel=[('coilL',i) for i in random.sample(coilL,3)]+[('coilR',i) for i in random.sample(coilR,2)]+[('wedge',i) for i in random.sample(wedge,5)]+[('thin',i) for i in random.sample([i for i in nc if 0.03<mxs(i)<0.07],2)]
def side(i):
    inf=E[i]['info'];return max('LRTB',key=lambda s:(inf.get('coil_'+s,0)*2+inf['strip_'+s]))
rows=[]
for kind,i in sel:
    m=idx[i];im=get_full(m);d=m['dpi'];s=150/d if d>150 else 1.0
    im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA) if s<1 else im;d*=s
    out,info=G.clean_edges(im,d,fill=(255,255,255))
    h,w=im.shape[:2];sd=side(i)
    def crop(x):
        if sd=='L': c=x[int(h*0.55):int(h*0.55)+330,:150]
        elif sd=='R': c=x[int(h*0.55):int(h*0.55)+330,w-150:]
        elif sd=='T': c=np.rot90(x[:150,int(w*0.3):int(w*0.3)+330])
        else: c=np.rot90(x[h-150:,int(w*0.3):int(w*0.3)+330])
        return cv2.resize(c,None,fx=2,fy=2,interpolation=cv2.INTER_NEAREST)
    th=lambda x:cv2.resize(x,(220,int(220*h/w)),interpolation=cv2.INTER_AREA)
    a,b=crop(im),crop(out)
    cell=np.hstack([th(im),th(out)])
    H=max(cell.shape[0],a.shape[0]);
    pad=lambda x:cv2.copyMakeBorder(x,0,H-x.shape[0],0,4,cv2.BORDER_CONSTANT,value=(120,120,120))
    row=np.hstack([pad(cell),pad(a),pad(b)])[:,:,::-1].copy()
    cv2.putText(row,f'{kind} #{i} side {sd} painted {info["painted_frac"]*100:.1f}%  before|after thumbs, edge crop 2x before, after',(3,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,220),1,cv2.LINE_AA)
    rows.append(row)
W=max(r.shape[1] for r in rows)
rows=[cv2.copyMakeBorder(r,0,0,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(255,255,255)) for r in rows]
cv2.imwrite('panels/edge_cleanup_A.png',np.vstack(rows[:5]));cv2.imwrite('panels/edge_cleanup_B.png',np.vstack(rows[5:]))
print([k for k,_ in sel])
