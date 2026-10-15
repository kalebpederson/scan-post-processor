import cv2, numpy as np
from skimage.metrics import structural_similarity as ssim
from lib import *
regs=[('epson_C_pencil (600 src)',cv2.rotate(cv2.imread(f'{SCANS}/epson_600dpi.png',0),cv2.ROTATE_90_CLOCKWISE) if False else None,600,(5216,200,7016,600)),]
e=cv2.rotate(cv2.imread(f'{SCANS}/epson_600dpi.png',0),cv2.ROTATE_90_CLOCKWISE)
R=[('epson_A light halftone',e,600,(3000,4380,4800,4780)),('epson_B heading+small',e,600,(840,4260,2640,4660)),('epson_C pencil',e,600,(5216,200,7016,600))]
for n,f,b in [('corpus_A smallest printed','single04.jpg',(500,2400,1400,3000)),('corpus_B handwritten exponent','single13.jpg',(1400,2280,2400,2610)),('corpus_C light pencil','pair03_back.jpg',(380,1850,1380,2200))]:
    R.append((n,cv2.imread(f'{CORPUS}/{f}',0),300,b))
print('| region | '+' | '.join(str(r) for r in RATES if r<300)+' |'); print('|---|'+'---|'*7)
for n,g,sd,(x0,y0,x1,y1) in R:
    row=[]
    ref=g[y0:y1,x0:x1]
    for r in RATES:
        if r>=sd: continue
        up=upsample(downsample(g,sd,r),(g.shape[1],g.shape[0]))[y0:y1,x0:x1]
        row.append(ssim(ref,up,data_range=255))
    print(f'| {n} | '+' | '.join(f'{v:.3f}' for v in row)+' |')
