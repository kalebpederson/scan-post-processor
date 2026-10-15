import json, sys, collections
from multiprocessing import Pool
from lib import *
PSM=int(os.environ.get('PSM','11'))
REF=json.load(open(f'{WORK}/consensus.json'))
def cls(w):
    d=w['dark']
    return 'bold' if d>=50 else ('mid' if d>=35 else 'light')
for w in REF: w['cls']=cls(w); w['bw']=(w['cls']=='bold' and w['text'].isalpha() and len(w['text'])>=4)
# group lines
def lines(ref):
    ws=sorted(ref,key=lambda w:(w['y']+w['h']/2))
    groups=[]
    for w in ws:
        cy=w['y']+w['h']/2
        for g in groups:
            if abs(g['cy']-cy)<0.5*w['h']:
                last=max(g['ws'],key=lambda v:v['x']+v['w'])
                first=min(g['ws'],key=lambda v:v['x'])
                if 0<=w['x']-(last['x']+last['w'])<60 or 0<=first['x']-(w['x']+w['w'])<60:
                    g['ws'].append(w); g['cy']=np.mean([v['y']+v['h']/2 for v in g['ws']]); break
        else: groups.append(dict(cy=cy,ws=[w]))
    return groups
LINES=lines(REF)
def lev(a,b):
    if not a: return len(b)
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1):
            cur.append(min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]
def score(hyp,scale):
    """hyp boxes in image px; scale = 600grid px / image px"""
    H=[dict(text=h['text'],cx=(h['x']+h['w']/2)*scale,cy=(h['y']+h['h']/2)*scale,hh=h['h']*scale) for h in hyp]
    # registration: hardware scans are separate scans (slightly different scale/offset). Fit x'=a x+b, y'=c y+d on
    # words whose text is unique in both ref and hyp, then map hyp centers onto the 600 grid before positional matching.
    rc0=collections.Counter(w['text'] for w in REF); hc0=collections.Counter(h['text'] for h in H)
    pr=[]
    for w in REF:
        if rc0[w['text']]==1 and hc0[w['text']]==1:
            h=[h for h in H if h['text']==w['text']][0]; pr.append((h['cx'],h['cy'],w['x']+w['w']/2,w['y']+w['h']/2))
    reg=None
    if len(pr)>=25:
        pr=np.array(pr); keep=np.ones(len(pr),bool)
        for _ in range(4):
            ax=np.polyfit(pr[keep,0],pr[keep,2],1); ay=np.polyfit(pr[keep,1],pr[keep,3],1)
            res=np.hypot(np.polyval(ax,pr[:,0])-pr[:,2],np.polyval(ay,pr[:,1])-pr[:,3]); keep=res<max(8,2.5*np.median(res[keep]))
            if keep.sum()<15: break
        for h in H: h['cx']=float(np.polyval(ax,h['cx'])); h['cy']=float(np.polyval(ay,h['cy']))
        reg=[float(ax[0]),float(ax[1]),float(ay[0]),float(ay[1]),int(keep.sum())]
    # multiset
    rc=collections.Counter(w['text'] for w in REF); hc=collections.Counter(h['text'] for h in H)
    ms=sum(min(rc[t],hc[t]) for t in rc)
    # positional
    used=set(); hit=[]
    for w in REF:
        cx,cy=w['x']+w['w']/2,w['y']+w['h']/2; best=None
        for j,h in enumerate(H):
            if j in used or h['text']!=w['text']: continue
            d=abs(h['cx']-cx)+abs(h['cy']-cy)
            if d<20 and (best is None or d<best[0]): best=(d,j)
        if best: used.add(best[1]); hit.append(True)
        else: hit.append(False)
    out=dict(n_hyp=len(H),ms_recall=ms/len(REF),reg=reg)
    out['pos_recall']=float(np.mean(hit))
    for c in ['bold','mid','light']:
        idx=[i for i,w in enumerate(REF) if w['cls']==c]
        out['pos_'+c]=float(np.mean([hit[i] for i in idx])); out['n_'+c]=len(idx)
    hd=[i for i,w in enumerate(REF) if w['bw']]
    out['pos_heading']=float(np.mean([hit[i] for i in hd])); out['n_heading']=len(hd)
    out['hits']=[int(x) for x in hit]
    # CER
    errs=tot=0
    for g in LINES:
        ws=sorted(g['ws'],key=lambda v:v['x']); rt=' '.join(v['text'] for v in ws)
        x0=min(v['x'] for v in ws)-10; x1=max(v['x']+v['w'] for v in ws)+10
        y0=min(v['y'] for v in ws)-0.3*ws[0]['h']; y1=max(v['y']+v['h'] for v in ws)+0.3*ws[0]['h']
        hw=sorted([h for h in H if x0<=h['cx']<=x1 and y0<=h['cy']<=y1],key=lambda h:h['cx'])
        ht=' '.join(h['text'] for h in hw)
        errs+=lev(rt,ht); tot+=len(rt)
    out['cer']=errs/tot
    return out
def load(src):  # src dpi of hardware scan
    return cv2.imread(f'{WORK}/gray_{src}.png',0)
def job(a):
    src,r=a
    g=load(src); res=dict(src=src,r=r)
    if r==src: im=g; res['t_down_ms']=0
    else:
        t=time.perf_counter(); im=downsample(g,src,r); res['t_down_ms']=(time.perf_counter()-t)*1e3
    res['shape']=list(im.shape)
    scale=7016/im.shape[1]
    res['direct']=score(ocr(im,r,PSM),scale)
    up=upsample(im,(7016,5096)) if im.shape[1]!=7016 else im
    res['rt']=score(ocr(up,600,PSM),1.0)
    return res
if __name__=='__main__':
    jobs=[]
    for src in [300,400,600,1200]:
        for r in RATES+[400,600]:
            if r<=src: jobs.append((src,r))
    # sizes at 600 ref for hw natives
    jobs=sorted(set(jobs),key=lambda j:-j[0])
    jobs+= [(150,150),(200,200)]
    out=[]
    with Pool(14) as p:
        for res in p.imap_unordered(job,jobs):
            out.append(res); print(res['src'],res['r'],round(res['rt']['ms_recall'],3),round(res['direct']['ms_recall'],3),flush=True)
    json.dump(dict(nref=len(REF),classes=collections.Counter(w['cls'] for w in REF),out=out),open(f'{HERE}/epson_results_psm{PSM}.json','w'),indent=1)
