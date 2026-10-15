import cv2,numpy as np
def sheet(items,cols=6,tw=300,out='x.png',label=True):
    """items: list of (img(BGR or gray), caption)"""
    tiles=[]
    for im,cap in items:
        if im.ndim==2: im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        h,w=im.shape[:2]; s=tw/w; th=int(h*s)
        t=cv2.resize(im,(tw,th),interpolation=cv2.INTER_AREA)
        t=cv2.copyMakeBorder(t,0,0,0,0,cv2.BORDER_CONSTANT)
        tiles.append((t,cap))
    rows=[tiles[i:i+cols] for i in range(0,len(tiles),cols)]
    H=[max(t.shape[0] for t,_ in r)+16 for r in rows]
    canvas=np.full((sum(H),cols*(tw+4)),200,np.uint8); canvas=np.dstack([canvas]*3)
    y=0
    for r,hh in zip(rows,H):
        for j,(t,cap) in enumerate(r):
            canvas[y+14:y+14+t.shape[0],j*(tw+4):j*(tw+4)+tw]=t
            cv2.putText(canvas,cap[:48],(j*(tw+4)+2,y+11),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,0,160),1,cv2.LINE_AA)
        y+=hh
    cv2.imwrite(out,canvas)
