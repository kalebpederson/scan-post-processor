"""Rotation interpolation comparison. Original straight page O (full-res) -> synthetic skew S (known angle, supersampled cubic,
paper-colour fill) -> correct S by angle -> compare with O (SSIM, MAE, edge sharpness)."""
import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,time
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim
from fullres import get_full
cv2.setNumThreads(1)
idx=json.load(open('index.json'));cand=json.load(open('gt_candidates.json'))
def rot(img,ang,method,fill):
    h,w=img.shape[:2];M=cv2.getRotationMatrix2D((w/2,h/2),ang,1.0)
    if method=='linear': return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR,borderValue=fill)
    if method=='cubic': return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderValue=fill)
    if method=='lanczos': return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LANCZOS4,borderValue=fill)
    if method=='ss2_area':    # 2x supersample (cubic) -> linear rotate -> INTER_AREA down
        big=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
        M2=cv2.getRotationMatrix2D((w,h),ang,2.0/2.0)  # centre in big coords
        M2=cv2.getRotationMatrix2D((w,h),ang,1.0)
        r=cv2.warpAffine(big,M2,(2*w,2*h),flags=cv2.INTER_LINEAR,borderValue=fill)
        return cv2.resize(r,(w,h),interpolation=cv2.INTER_AREA)
    if method=='nearest': return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_NEAREST,borderValue=fill)
METHODS=['nearest','linear','cubic','lanczos','ss2_area']
def sharp(g):  # mean gradient magnitude of top 2% edge pixels
    gx=cv2.Sobel(g,cv2.CV_32F,1,0);gy=cv2.Sobel(g,cv2.CV_32F,0,1);m=np.hypot(gx,gy)
    return float(np.percentile(m,99))
def work(t):
    i,ang=t
    O=get_full(idx[i])                       # RGB uint8 full-res
    fill=tuple(int(v) for v in np.percentile(O.reshape(-1,3),90,axis=0))
    S=rot(O,ang,'ss2_area',fill)             # synthetic skew (+ang)
    out={}
    for m in METHODS:
        t0=time.perf_counter(); C=rot(S,-ang,m,fill); dt=time.perf_counter()-t0
        h,w=O.shape[:2];my,mx=int(h*0.12),int(w*0.12)   # central crop avoids filled corners
        a=cv2.cvtColor(O,cv2.COLOR_RGB2GRAY)[my:-my,mx:-mx];b=cv2.cvtColor(C,cv2.COLOR_RGB2GRAY)[my:-my,mx:-mx]
        out[m]=dict(ssim=float(ssim(a,b,data_range=255)),mae=float(np.abs(a.astype(float)-b).mean()),sharp=sharp(b),ms=dt*1000)
    out['ref_sharp']=sharp(cv2.cvtColor(O,cv2.COLOR_RGB2GRAY)[my:-my,mx:-mx])
    out['dpi']=idx[i]['dpi']
    return dict(i=i,ang=ang,res=out)
if __name__=='__main__':
    rs=np.random.RandomState(77)
    sel=[i for i in cand if abs(idx[i]['dpi']-200)<5][:10]+[i for i in cand if abs(idx[i]['dpi']-300)<5][:6]
    tr=[(i,float(rs.choice([-1,1])*rs.uniform(0.3,2.5))) for i in sel for _ in range(2)]
    with Pool(8) as p: R=p.map(work,tr)
    json.dump(R,open('rot_quality.json','w'))
    for m in METHODS:
        for name,flt in (('all',lambda r:True),('200dpi',lambda r:r['res']['dpi']<250),('300dpi',lambda r:r['res']['dpi']>=250)):
            rr=[r for r in R if flt(r)]
            print(f"{m:9s} {name:7s} n={len(rr):2d} SSIM {np.mean([r['res'][m]['ssim'] for r in rr]):.4f} MAE {np.mean([r['res'][m]['mae'] for r in rr]):.3f} sharp/ref {np.mean([r['res'][m]['sharp']/r['res']['ref_sharp'] for r in rr]):.3f} ms {np.median([r['res'][m]['ms'] for r in rr]):.0f}")
