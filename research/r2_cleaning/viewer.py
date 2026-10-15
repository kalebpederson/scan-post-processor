"""crop viewer with native-coordinate ticks: viewer.py <img path or corpus name> y0 y1 x0 x1 out [scale] [stretch_lo]"""
import sys; sys.path.insert(0,'.')
from common import *
def view(rgb, y0,y1,x0,x1, scale=1.0, lo=0, tick=None):
    c=rgb[y0:y1,x0:x1].astype(np.float32)
    if lo: c=np.clip((c-lo)/(255-lo),0,1)*255
    c=c.astype(np.uint8)
    im=cv2.resize(c,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_NEAREST)
    h,w=im.shape[:2]
    tick=tick or int(max(50,round((x1-x0)/8/50)*50))
    for x in range((x0//tick+1)*tick,x1,tick):
        X=int((x-x0)*scale); cv2.line(im,(X,0),(X,h),(0,160,255),1); cv2.putText(im,str(x),(X+2,12),0,0.45,(255,0,255),1)
    for y in range((y0//tick+1)*tick,y1,tick):
        Y=int((y-y0)*scale); cv2.line(im,(0,Y),(w,Y),(0,160,255),1); cv2.putText(im,str(y),(2,Y-2),0,0.45,(255,0,255),1)
    return im
if __name__=="__main__":
    a=sys.argv; name=a[1]
    path=name if '/' in name else str(CORPUS/(name+'.jpg')) if (CORPUS/(name+'.jpg')).exists() else str(CORPUS/(name+'.png'))
    rgb=load_rgb(path); h,w=rgb.shape[:2]
    y0,y1,x0,x1=[int(v) if int(v)>=0 else 0 for v in a[2:6]]
    y1=min(y1,h) or h; x1=min(x1,w) or w
    scale=float(a[7]) if len(a)>7 else 1.0; lo=int(a[8]) if len(a)>8 else 0
    save_rgb(a[6], view(rgb,y0,y1,x0,x1,scale,lo))
