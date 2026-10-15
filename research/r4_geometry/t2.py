import json,cv2,numpy as np
from methods import ink_map,_rot
idx=json.load(open('index.json'))
im=cv2.imread(idx[51]['fn']);h,w=im.shape[:2]
for a in (-2.89,3.39):
    M=cv2.getRotationMatrix2D((w/2,h/2),a,1.0)
    r=cv2.warpAffine(im,M,(w,h),borderValue=(255,255,255));g=cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    ink=ink_map(g);hh,ww=ink.shape
    ink=ink[int(hh*.03):int(hh*.97),int(ww*.03):int(ww*.97)]
    ink=np.where(ink>.12,ink,0).astype(np.float32)
    angs=np.arange(a-2.5,a+2.5,0.25)
    v=[np.var(_rot(ink,x).sum(1)) for x in angs]
    v2=[np.var((_rot((ink>0).astype(np.float32),x)).sum(1)) for x in angs]
    d=[np.sum(np.diff(_rot(ink,x).sum(1))**2) for x in angs]
    for x,p,q,s in zip(angs,v,v2,d): print(a,round(x,2),int(p),int(q),int(s))
