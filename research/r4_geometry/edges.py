"""Section C: edge strips and hole analysis on 100 ppi cache."""
import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np
from multiprocessing import Pool
cv2.setNumThreads(1)
DPI=100
def paper_level(g):
    h,w=g.shape; c=g[h//4:3*h//4,w//4:3*w//4]
    return float(np.percentile(c,90))
def strip_profile(g,side,pl,frac=0.6,maxd=60):
    """run length (px) of dark pixels from the edge inward, per position along the edge"""
    if side=='L': a=g[:, :maxd]
    elif side=='R': a=g[:, ::-1][:, :maxd]
    elif side=='T': a=g[:maxd, :].T
    else: a=g[::-1, :][:maxd, :].T
    dark=a<frac*pl            # (positions, depth)
    # run from edge
    run=np.argmin(dark,axis=1)       # first non-dark index
    allk=dark.all(1); run=np.where(allk,maxd,run)
    return run
def smooth_max(run,win=25):
    k=np.ones(win)/win
    s=np.convolve(run,k,mode='valid') if len(run)>=win else run
    return float(s.max())
def holes(g,pl):
    """dark round blobs within 0.9in of L/R edges, not touching edge"""
    h,w=g.shape; out={}
    dark=(g<0.55*pl).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(dark,connectivity=8)
    res={'L':[], 'R':[]}
    for i in range(1,n):
        x,y,ww,hh,ar=st[i]
        if ar<25 or ar>1500 or ww>60 or hh>60: continue
        if x<=1 or y<=1 or x+ww>=w-1 or y+hh>=h-1: continue
        fill=ar/(ww*hh)
        if fill<0.5 or max(ww,hh)/max(1,min(ww,hh))>2.5: continue
        cx=x+ww/2
        if cx<0.9*DPI: res['L'].append((int(cx),int(y+hh/2),int(ar)))
        elif cx>w-0.9*DPI: res['R'].append((int(cx),int(y+hh/2),int(ar)))
    return res
def work(m):
    g=cv2.imread(m['fn'],0); pl=paper_level(g)
    r={'paper':pl}
    for s in 'LRTB':
        run=strip_profile(g,s,pl)
        r['run_'+s]=smooth_max(run); r['cov_'+s]=float((run>0).mean()); r['med_'+s]=float(np.median(run))
        # wedge slope: linear fit of run vs position where run>0
        pos=np.arange(len(run)); k=run>=2
        if k.sum()>0.2*len(run):
            p=np.polyfit(pos[k],run[k],1); r['slope_'+s]=float(np.degrees(np.arctan(p[0])))
        else: r['slope_'+s]=None
    r['holes']=holes(g,pl)
    return r
if __name__=='__main__':
    idx=json.load(open('index.json'))
    with Pool(10) as p: R=p.map(work,idx,chunksize=8)
    json.dump(R,open('edges_raw.json','w'))
