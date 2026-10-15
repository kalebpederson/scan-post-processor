import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,sys
from multiprocessing import Pool
from fullres import get_full
cv2.setNumThreads(1)
idx=json.load(open('index.json'));C=json.load(open('comp_raw.json'));S=json.load(open('label_set.json'))
ink=json.load(open('ink_raw.json'));fc=np.array([x['f0.1'] for x in ink])
T=150
def tiles(i):
    m=idx[i];W,H=m['sw'],m['sh']
    cs=[c for c in C[i][0] if c['peak']>=0.15 and c['area']>=8 and c['x']>=3 and c['y']>=3 and c['x']+c['w']<=W-3 and c['y']+c['h']<=H-3]
    cs.sort(key=lambda c:-(c['peak']*np.sqrt(c['area'])))
    cs=cs[:2]
    out=[]
    if not cs: return i,[(np.full((T,T,3),255,np.uint8),'no cand')]
    im=get_full(m)[:,:,::-1]; s=im.shape[1]/W
    half=int(0.375*(m['w'] and im.shape[1]/(8.5) or 200)) # 0.75in square
    for c in cs:
        cx=int((c['x']+c['w']/2)*s);cy=int((c['y']+c['h']/2)*s)
        x0=min(max(cx-half,0),im.shape[1]-2*half);y0=min(max(cy-half,0),im.shape[0]-2*half)
        crop=im[y0:y0+2*half,x0:x0+2*half]
        crop=cv2.resize(crop,(T,T),interpolation=cv2.INTER_AREA if 2*half>T else cv2.INTER_CUBIC)
        out.append((crop,f"{i} p{c['peak']:.2f} a{c['area']}"))
    return i,out
if __name__=='__main__':
    from sheet import sheet
    with Pool(12) as p: R=p.map(tiles,S,chunksize=4)
    items=[]
    nocand=[i for i,t in R if t[0][1]=='no cand']
    json.dump(nocand,open('nocand.json','w'));print('nocand',len(nocand))
    for i,t in R:
        if t[0][1]=='no cand': continue
        items+=t
        if len(t)==1: items.append((np.full((T,T,3),255,np.uint8),''))
    per=40
    for s in range(0,len(items),per):
        sheet(items[s:s+per],cols=8,tw=T,out=f'crops_{s//per:02d}.png')
    print(len(items))
