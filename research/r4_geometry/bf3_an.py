import json,numpy as np
idx=json.load(open('index.json'));F=json.load(open('blank_feats3.json'))
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
def sc(fn): return np.array([fn(F[i]) for i in range(len(idx))])
cands={'max mass':lambda f:max([c['mass'] for c in f],default=0),'total mass':lambda f:sum(c['mass'] for c in f)}
for g0 in (20,30,45,60):
    cands[f'max ms{g0}']=(lambda g0:lambda f:max([c[f'ms{g0}'] for c in f],default=0))(g0)
    cands[f'total ms{g0}']=(lambda g0:lambda f:sum(c[f'ms{g0}'] for c in f))(g0)
for k,fn in cands.items():
    s=sc(fn);b=s[blank];nb=s[~blank]
    print(f"{k:14s} BLANK max {b.max():7.1f} p99 {np.percentile(b,99):6.1f} | NB min {nb.min():7.1f} p1 {np.percentile(nb,1):7.1f} | ratio NBmin/BLANKmax {nb.min()/max(b.max(),1e-9):.2f}")
