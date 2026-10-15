import sys, json; sys.path.insert(0,'.')
from common import *
def montage(items, out, cols=3, lo=0, scale=1.0, pad=6):
    """items: list of (rgb_full, (x,y,w,h), caption)"""
    tiles=[]
    for rgb,(x,y,w,h),cap in items:
        c=rgb[y:y+h,x:x+w].astype(np.float32)
        if lo: c=np.clip((c-lo)/(255-lo),0,1)*255
        c=c.astype(np.uint8)
        if scale!=1: c=cv2.resize(c,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST if scale>1 else cv2.INTER_AREA)
        c=cv2.copyMakeBorder(c,18,pad,pad,pad,cv2.BORDER_CONSTANT,value=(128,128,128))
        cv2.putText(c,cap,(4,13),0,0.42,(255,255,0),1)
        tiles.append(c)
    Wm=max(t.shape[1] for t in tiles); Hm=max(t.shape[0] for t in tiles)
    rows=[]
    for i in range(0,len(tiles),cols):
        row=[cv2.copyMakeBorder(t,0,Hm-t.shape[0],0,Wm-t.shape[1],cv2.BORDER_CONSTANT,value=(128,128,128)) for t in tiles[i:i+cols]]
        while len(row)<cols: row.append(np.full((Hm,Wm,3),128,np.uint8))
        rows.append(np.concatenate(row,1))
    save_rgb(out,np.concatenate(rows,0))
