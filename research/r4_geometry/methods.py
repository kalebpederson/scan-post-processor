"""Candidate skew estimators for the study. All take uint8 gray at ~100 ppi (any size) and
return (angle_deg, confidence). angle: CCW-positive skew of the content as seen on screen
(text baseline rising to the right = positive)."""
import numpy as np, cv2
cv2.setNumThreads(1)

def ink_map(gray, dpi=100):
    """Background-normalised ink density in [0,1] (0 = paper)."""
    g = gray.astype(np.float32)
    k = int(round(dpi*0.25))|1
    bg = cv2.dilate(g, np.ones((3,3),np.uint8))
    bg = cv2.medianBlur(cv2.dilate(gray,np.ones((k,k),np.uint8)), 5).astype(np.float32)  # local paper level
    bg = cv2.GaussianBlur(bg,(0,0),k/3)
    n = g/np.maximum(bg,1)
    return np.clip(1-n,0,1)

def _rot(img, a, interp=cv2.INTER_LINEAR):
    h,w=img.shape[:2]
    M=cv2.getRotationMatrix2D((w/2,h/2),-a,1.0)   # correction by -a
    return cv2.warpAffine(img,M,(w,h),flags=interp,borderValue=0)

def _peak_hist(angles, weights, lo, hi, bin_=0.1, sm=0.5):
    """weighted histogram peak (deg) after gaussian smoothing; returns (peak, frac_weight_within_0.3deg)"""
    if len(angles)==0: return 0.0,0.0
    nb=int(round((hi-lo)/bin_))
    h,e=np.histogram(angles,bins=nb,range=(lo,hi),weights=weights)
    s=int(round(sm/bin_))
    ker=np.exp(-0.5*(np.arange(-3*s,3*s+1)/s)**2); ker/=ker.sum()
    hs=np.convolve(h,ker,mode='same')
    i=int(np.argmax(hs)); pk=(e[i]+e[i+1])/2
    m=np.abs(angles-pk)<0.3
    return float(pk), float(weights[m].sum()/max(weights.sum(),1e-9))

# ---------- (a) projection profile
def est_projection(gray, dpi=100, rng=12.0):
    ink = ink_map(gray, dpi)
    h,w=ink.shape
    ink = ink[int(h*.03):int(h*.97), int(w*.03):int(w*.97)]
    ink = np.where(ink>0.12, ink, 0).astype(np.float32)   # suppress faint bleed/noise
    def score(img,a):
        r=_rot(img,a).sum(1)
        return float(np.var(r))
    sm=cv2.resize(ink,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    angs=np.arange(-rng,rng+1e-9,0.5)
    sc=np.array([score(sm,a) for a in angs])
    a0=angs[sc.argmax()]
    angs2=np.arange(a0-0.6,a0+0.6+1e-9,0.1)
    sc2=np.array([score(ink,a) for a in angs2])
    a1=angs2[sc2.argmax()]
    angs3=np.arange(a1-0.1,a1+0.1+1e-9,0.02)
    sc3=np.array([score(ink,a) for a in angs3])
    best=angs3[sc3.argmax()]
    # confidence: peak vs median of coarse scores, and sharpness
    ia=int(sc.argmax()); nb=[j for j in range(len(angs)) if 1.0<=abs(angs[j]-angs[ia])<=2.5]
    conf=float((sc[ia]-np.mean(sc[nb]))/(sc[ia]+1e-9)) if nb else 0.0
    return float(best), conf

# ---------- (b) Hough
def est_hough(gray, dpi=100, rng=12.0):
    if dpi>75:
        gray=cv2.resize(gray,None,fx=75/dpi,fy=75/dpi,interpolation=cv2.INTER_AREA); dpi=75
    g=cv2.GaussianBlur(gray,(0,0),0.6)
    e=cv2.Canny(g,40,110)
    minlen=int(dpi*1.2)
    L=cv2.HoughLinesP(e,1,np.pi/1800,threshold=int(dpi*0.7),minLineLength=minlen,maxLineGap=int(dpi*0.08))
    if L is None: return 0.0,0.0
    L=L.reshape(-1,4).astype(np.float32)
    dx=L[:,2]-L[:,0]; dy=L[:,3]-L[:,1]
    ang=np.degrees(np.arctan2(-dy,dx)); ang=(ang+90)%180-90      # (-90,90]
    ln=np.hypot(dx,dy)
    # horizontals and verticals both (vertical: deviation from +-90)
    dev=np.where(np.abs(ang)<=45,ang,ang-np.sign(ang)*90)
    m=np.abs(dev)<=rng
    pk,fr=_peak_hist(dev[m],ln[m],-rng,rng)
    return pk, fr*min(1.0,m.sum()/10)

# ---------- (c) connected components
def _blobs(gray,dpi):
    ink=ink_map(gray,dpi)
    b=(ink>0.25).astype(np.uint8)
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(max(3,int(dpi*0.12)),1))
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k)
    return b
def est_cc_rect(gray, dpi=100, rng=12.0):
    b=_blobs(gray,dpi)
    cs,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    A=[];W=[]
    for c in cs:
        if len(c)<20: continue
        (cx,cy),(w,h),a=cv2.minAreaRect(c)
        if max(w,h)<dpi*0.35 or max(w,h)/max(min(w,h),1)<2.5 or min(w,h)>dpi*0.5: continue
        if w<h: a=a+90   # opencv angle convention -> angle of long side
        a=-((a+90)%180-90)   # to CCW-positive
        a=(a+90)%180-90
        A.append(a);W.append(max(w,h))
    A=np.array(A);W=np.array(W)
    m=np.abs(A)<=rng
    pk,fr=_peak_hist(A[m],W[m],-rng,rng,0.25,0.6)
    return pk,fr*min(1.0,m.sum()/15)
def est_cc_moments(gray, dpi=100, rng=12.0):
    b=_blobs(gray,dpi)
    n,lab,st,cen=cv2.connectedComponentsWithStats(b,connectivity=8)
    A=[];W=[]
    for i in range(1,n):
        x,y,w,h,ar=st[i]
        if w<dpi*0.35 or w>dpi*4 or h>dpi*0.5: continue
        m=cv2.moments((lab[y:y+h,x:x+w]==i).astype(np.uint8),binaryImage=True)
        mu20,mu02,mu11=m['mu20'],m['mu02'],m['mu11']
        d=np.sqrt(4*mu11**2+(mu20-mu02)**2)
        l1=(mu20+mu02+d)/2; l2=(mu20+mu02-d)/2
        if l2<=0 or l1/l2<8: continue
        th=0.5*np.arctan2(2*mu11,mu20-mu02)   # image coords, y down
        A.append(-np.degrees(th)); W.append(w)
    A=np.array(A);W=np.array(W)
    if len(A)==0: return 0.0,0.0
    m=np.abs(A)<=rng
    pk,fr=_peak_hist(A[m],W[m],-rng,rng,0.25,0.6)
    return pk,fr*min(1.0,m.sum()/15)

# ---------- (e) gradient orientation (mod 90)
def est_gradient(gray, dpi=100, rng=12.0):
    g=cv2.GaussianBlur(gray,(0,0),1.0).astype(np.float32)
    gx=cv2.Scharr(g,cv2.CV_32F,1,0); gy=cv2.Scharr(g,cv2.CV_32F,0,1)
    mag=np.hypot(gx,gy)
    m=mag>np.percentile(mag,97)
    m&=mag>60
    ang=np.degrees(np.arctan2(gy[m],gx[m]))   # gradient direction; edge dir = +90
    # horizontal edge: gradient ~ +-90 ; deviation d from nearest axis
    d=(ang+45)%90-45         # gradient dev from x-axis in (-45,45]
    # for horizontal text lines, gradient is vertical; angle of gradient y-down.. combine both axes
    # edge rotated CCW by t (screen) -> gradient vector rotated CCW by t on screen = -t in image coords
    dev=-d
    w=mag[m]
    k=np.abs(dev)<=rng
    pk,fr=_peak_hist(dev[k],w[k],-rng,rng,0.1,0.4)
    return pk,fr

# ---------- (f) LSD line segments
_lsd=None
def est_lsd(gray, dpi=100, rng=12.0):
    global _lsd
    if _lsd is None: _lsd=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    L=_lsd.detect(gray)[0]
    if L is None: return 0.0,0.0
    L=L.reshape(-1,4)
    dx=L[:,2]-L[:,0]; dy=L[:,3]-L[:,1]
    ln=np.hypot(dx,dy); k=ln>=dpi*0.3
    dx,dy,ln=dx[k],dy[k],ln[k]
    ang=np.degrees(np.arctan2(-dy,dx)); ang=(ang+90)%180-90
    dev=np.where(np.abs(ang)<=45,ang,ang-np.sign(ang)*90)
    m=np.abs(dev)<=rng
    pk,fr=_peak_hist(dev[m],ln[m]**2,-rng,rng,0.1,0.35)
    return pk, fr*min(1.0,m.sum()/10)

def _cropped(f):
    def g(gray,dpi=100,**kw):
        h,w=gray.shape
        mx=int(round(0.5*dpi)); my=int(round(0.25*dpi))
        return f(gray[my:h-my,mx:w-mx],dpi,**kw)
    g.__name__=f.__name__
    return g
METHODS_RAW={'proj':est_projection,'hough':est_hough,'cc_rect':est_cc_rect,'cc_mom':est_cc_moments,'grad':est_gradient,'lsd':est_lsd}

METHODS={k:_cropped(v) for k,v in METHODS_RAW.items()}
