import cv2, numpy as np, json
from lib import *
cv2.setNumThreads(8)
OUT=f'{HERE}/panels'
VIEWS=[('110 ppi laptop @100%',110),('160 ppi tablet @100%',160),('110 ppi laptop @200% (=220 ppi)',220)]
def toscreen(tile,scale):
    h,w=tile.shape[:2]; size=(max(1,round(w*scale)),max(1,round(h*scale)))
    return cv2.resize(tile,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
def label(img,text,h=22,fs=0.5):
    bar=np.full((h,img.shape[1],3),255,np.uint8); cv2.putText(bar,text,(3,h-7),cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,160),1,cv2.LINE_AA)
    return np.vstack([bar,img])
def panel(name,title,page,src_dpi,box,rates,extra_src=True):
    x0,y0,x1,y1=box; rows=[]
    rl=[('source %d ppi'%src_dpi,src_dpi)] if extra_src else []
    rl+= [('%d ppi'%r,r) for r in rates if r<src_dpi or not extra_src]
    for lab,r in rl:
        d=page if r==src_dpi else downsample(page,src_dpi,r)
        f=r/src_dpi; crop=d[round(y0*f):round(y1*f),round(x0*f):round(x1*f)]
        cells=[]
        for vn,sp in VIEWS:
            t=toscreen(crop,sp/r); cells.append(label(t,'%s | %s'%(lab,vn.split(' ')[0]+' '+vn.split(' ')[1]) if False else '%s -> %s'%(lab,vn),fs=0.4))
        H=max(c.shape[0] for c in cells); 
        cells=[np.pad(c,((0,H-c.shape[0]),(0,6),(0,0)),constant_values=255) for c in cells]
        rows.append(np.hstack(cells))
    W=max(r.shape[1] for r in rows)
    rows=[np.pad(r,((0,4),(0,W-r.shape[1]),(0,0)),constant_values=255) for r in rows]
    img=np.vstack(rows); img=label(img,title,h=26,fs=0.55)
    assert img.shape[1]<=2000,img.shape
    cv2.imwrite(f'{OUT}/{name}.png',img); print(name,img.shape)
def epson(): 
    im=cv2.imread(f'{SCANS}/epson_600dpi.png'); return cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
if __name__=='__main__':
    e=epson()
    panel('epson_A_light_halftone_small_text','Epson 600-src: lightest halftone small text (~6pt), area-downsampled to r then viewed',e,600,(3000,4380,4800,4780),RATES+[400])
    panel('epson_B_heading_plus_small','Epson 600-src: bold heading + small gray text',e,600,(840,4260,2640,4660),RATES+[400])
    panel('epson_C_pencil_handwriting','Epson 600-src: pencil handwriting on notebook page',e,600,(5216,200,7016,600),RATES+[400])
    for nm,f,box,ti in [('corpus_A_smallest_printed','single04.jpg',(500,2400,1400,3000),'Corpus single04 (300dpi): smallest printed text (definition captions)'),
                        ('corpus_B_handwritten_exponent','single13.jpg',(1400,2280,2400,2610),'Corpus single13 (300dpi): handwritten exponents / radicals'),
                        ('corpus_C_light_pencil','pair03_back.jpg',(380,1850,1380,2200),'Corpus pair03_back (300dpi): light pencil handwriting')]:
        g=cv2.imread(f'{CORPUS}/{f}'); panel(nm,ti,g,300,box,RATES)
