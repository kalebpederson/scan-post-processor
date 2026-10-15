import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
import geometry as G
from fullres import get_full
from sheet import sheet
cv2.setNumThreads(1)
idx=json.load(open('index.json'));R=json.load(open('final_survey.json'))
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
ang=np.array([r['ang'] for r in R]);conf=np.array([r['conf'] for r in R])
cand=[i for i in np.argsort(-np.abs(ang)) if conf[i]>=0.6 and not blank[i]][:12]
def grid(img,step=40):
    o=img.copy()
    for y in range(0,o.shape[0],step): o[y,:]=(o[y,:]*0.55+np.array([0,0,255])*0.45).astype(np.uint8)
    return o
def w(i):
    m=idx[i];im=get_full(m);d=m['dpi']
    s=min(1.0,150/d);im=cv2.resize(im,None,fx=s,fy=s,interpolation=cv2.INTER_AREA) if s<1 else im;d=d*s
    a=ang[i];out=G.deskew(im,a,fill=G.paper_colour(im))
    h,wd=im.shape[:2]
    def th(x): return cv2.resize(x,(380,int(380*h/wd)),interpolation=cv2.INTER_AREA)
    top=np.hstack([th(im),np.full((th(im).shape[0],6,3),90,np.uint8),th(out)])
    y0=int(h*0.12);y1=y0+260
    crop=lambda x:grid(x[y0:y1,int(wd*0.05):int(wd*0.05)+380])
    bot=np.hstack([crop(im),np.full((260,6,3),90,np.uint8),crop(out)])
    img=np.vstack([top,bot])[:,:,::-1].copy()
    name=idx[i]['pdf'].split('/')[-1][:50]
    cv2.putText(img,f'#{i} {name} p{idx[i]["page"]+1} est={a:+.2f} deg conf={conf[i]:.2f}  BEFORE | AFTER',(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,200),1,cv2.LINE_AA)
    fn=f'panels/skew_{i:04d}_{a:+.2f}.png';cv2.imwrite(fn,img);return fn,img
if __name__=='__main__':
    with Pool(6) as p: R2=p.map(w,[int(i) for i in cand])
    print([f for f,_ in R2])
    json.dump([f for f,_ in R2],open('panels_skew.json','w'))
