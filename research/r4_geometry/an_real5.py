import json,numpy as np
idx=json.load(open('index.json'));A=np.load('A7.npy');C=np.load('C7.npy')
N=['proj','hough','cc_rect','cc_mom','grad','lsd'];ji={n:j for j,n in enumerate(N)}
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))]);nb=~blank
def vote(names,tol,need):
    M=A[:,[ji[n] for n in names]];out=np.zeros(len(M));sup=np.zeros(len(M),int)
    for k in range(len(M)):
        best=None
        for j in range(M.shape[1]):
            s=np.abs(M[k]-M[k,j])<tol
            if best is None or s.sum()>best.sum(): best=s
        out[k]=M[k,best].mean();sup[k]=best.sum()
    return out,sup
for names in (['proj','lsd','cc_rect','grad'],['proj','lsd','cc_rect','grad','hough'],['proj','lsd','cc_rect','grad','hough','cc_mom']):
    o,s=vote(names,0.3,0)
    print(names,'support hist (nonblank):',np.bincount(s[nb],minlength=len(names)+1))
    for need in (2,3,4):
        rel=nb&(s>=need);a=np.abs(o[rel])
        print(f'  need>={need}: reliable {rel.sum()} ({rel.sum()/nb.sum()*100:.1f}%) |a| p50 {np.percentile(a,50):.2f} p90 {np.percentile(a,90):.2f} p99 {np.percentile(a,99):.2f} max {a.max():.2f}; >0.25: {(a>0.25).sum()} >0.5: {(a>0.5).sum()} >1: {(a>1).sum()} >2: {(a>2).sum()}')
o,s=vote(['proj','lsd','cc_rect','grad'],0.3,0)
np.save('vote4.npy',np.c_[o,s])
