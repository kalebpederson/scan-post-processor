import json,sys,random,pickle,numpy as np,cv2,pikepdf
from multiprocessing import Pool
from PIL import Image,ImageDraw,ImageFont
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/brother_live/work'; _os.makedirs(_WORK, exist_ok=True)
sys.path.insert(0,_ROOT + '/src')
from scanpp import geometry, pipeline, pdfio
S=_WORK
R=_ROOT + '/research/r4_geometry/'
I=json.load(open(R+'index.json'));L=json.load(open(R+'labels_blank.json'))['labels']
CROPS=pickle.load(open(S+'crops.pkl','rb'));CROPS[850]=(CROPS[850][0][:,:200],CROPS[850][1])
D={'light':60,'dark':135,'pen':210}
FONT=['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']
def font(pt,dpi):
    for f in FONT:
        try:return ImageFont.truetype(f,int(round(pt/72*dpi*4)))
        except Exception:pass
    raise SystemExit('no font')
def blur(a,dpi):return cv2.GaussianBlur(a,(0,0),0.5*dpi/200)
def place(shape,a,x,y):
    c=np.zeros(shape,np.float32);h,w=a.shape;c[y:y+h,x:x+w]=a[:shape[0]-y,:shape[1]-x];return c
def line_alpha(shape,dpi,length_in,rng,width_in=0.013):
    H,W=shape;ss=4;c=np.zeros((H*ss//4,W*ss//4),np.uint8)  # draw at native then blur; use float AA via 4x ROI
    L_=length_in*dpi;ang=np.deg2rad(rng.uniform(-8,8));x0=rng.uniform(0.8*dpi,W-0.8*dpi-L_-10);y0=rng.uniform(1.0*dpi,H-1.0*dpi)
    big=np.zeros((H*2,W*2),np.uint8) if False else None
    # AA line via supersampled ROI
    pad=20;x1=x0+L_*np.cos(ang);y1=y0+L_*np.sin(ang)
    xa,xb=int(min(x0,x1))-pad,int(max(x0,x1))+pad;ya,yb=int(min(y0,y1))-pad,int(max(y0,y1))+pad
    roi=np.zeros(((yb-ya)*ss,(xb-xa)*ss),np.uint8)
    pts=[]
    n=max(2,int(L_/10))
    for k in range(n+1):
        t=k/n;px=x0+(x1-x0)*t;py=y0+(y1-y0)*t+rng.normal(0,0.3)*(1 if k not in(0,n) else 0)
        pts.append(((px-xa)*ss,(py-ya)*ss))
    cv2.polylines(roi,[np.array(pts,np.int32)],False,255,max(1,int(round(width_in*dpi*ss))),cv2.LINE_8)
    a=cv2.resize(roi,((xb-xa),(yb-ya)),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    out=np.zeros(shape,np.float32);out[ya:yb,xa:xb]=a;return out
def text_alpha(shape,dpi,txt,pt,rng):
    f=font(pt,dpi);im=Image.new('L',(int(len(txt)*pt/72*dpi*4)+80,int(pt/72*dpi*4*1.6)),0);ImageDraw.Draw(im).text((10,5),txt,fill=255,font=f)
    a=np.asarray(im).astype(np.float32)/255;a=cv2.resize(a,(a.shape[1]//4,a.shape[0]//4),interpolation=cv2.INTER_AREA)
    ys,xs=np.where(a>0.02);a=a[ys.min():ys.max()+1,xs.min():xs.max()+1]
    H,W=shape;x=int(rng.uniform(0.9*dpi,W-0.9*dpi-a.shape[1]));y=int(H-rng.uniform(0.3,0.9)*dpi-a.shape[0])
    return place(shape,a,x,y)
def apply(page,alpha,depth):
    a=blur(alpha,200)[...,None]
    return np.clip(page.astype(np.float32)-a*depth*(page.astype(np.float32)/255),0,255).astype(np.uint8)
def score(rgb,dpi):
    img=rgb.copy();out,info,g,d=geometry._clean_edges_g(img,dpi,inplace=True)
    return geometry.blank_score(g if g is not None else out,d if g is not None else dpi)
def work(args):
    i,seed=args;rng=np.random.default_rng(seed);e=I[i]
    pdf=pikepdf.open(_ROOT + '/'+e['pdf']);t=next(pdfio.iter_page_inputs(pdf,[e['page']]))
    page=pipeline._decode(t);dpi=t.dpi;H,W=page.shape[:2];res={'base':score(page,dpi)}
    def put(key,alpha,depth):
        res[key]=score(apply(page,alpha,depth),dpi)
    for k,(a,cdpi) in CROPS.items():
        for sc in(1.0,0.7):
            aa=cv2.resize(a,None,fx=sc*dpi/cdpi,fy=sc*dpi/cdpi,interpolation=cv2.INTER_AREA) if abs(sc*dpi/cdpi-1)>1e-3 else a
            x=int(rng.uniform(0.9*dpi,W-0.9*dpi-aa.shape[1]));y=int(rng.uniform(1.0*dpi,H-1.0*dpi-aa.shape[0]))
            al=place(page.shape[:2],aa,x,y)
            for m,dp in D.items():put(f'word|{m}|{"full" if sc==1 else "small"}|{k}',al,dp)
    for dig in('7','12'):
        for dp,nm in((220,'black'),(130,'gray')):
            for pt in(10,12):
                put(f'num|{dig}|{nm}|{pt}pt',text_alpha(page.shape[:2],dpi,dig,pt,rng),dp)
    put('num|Page 2 of 2|black|10pt',text_alpha(page.shape[:2],dpi,'Page 2 of 2',10,rng),220)
    for ln in(0.1,0.5,1,2):
        for m,dp in D.items():put(f'line|{m}|{ln}',line_alpha(page.shape[:2],dpi,ln,rng),dp)
    return i,res
if __name__=='__main__':
    bl=[int(k) for k,v in L.items() if v['label']=='BLANK' and I[int(k)]['dpi']>190]
    random.seed(5);ids=random.sample(bl,24)
    with Pool(12) as p:out=p.map(work,[(i,100+n) for n,i in enumerate(ids)])
    json.dump(out,open(S+'calib.json','w'))
