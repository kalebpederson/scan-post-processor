import json,numpy as np
idx=json.load(open('index.json'));F=json.load(open('blank_feats.json'))
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
def top(i,key): 
    return max([c[key] for c in F[i]],default=0.0)
def diag(c): return np.hypot(c['w'],c['h'])
feats={'mass':lambda i:top(i,'mass'),'npx':lambda i:top(i,'npx'),'diag':lambda i:max([diag(c) for c in F[i]],default=0),
 'peak':lambda i:top(i,'peak'),
 'mass_sharp':lambda i:max([c['mass'] for c in F[i] if c['grad']>40],default=0)}
for k,f in feats.items():
    s=np.array([f(i) for i in range(len(idx))])
    b=s[blank];nb=s[~blank]
    print(f"{k:10s} BLANK max {b.max():.1f} p99 {np.percentile(b,99):.1f} p90 {np.percentile(b,90):.1f} | NONBLANK min {nb.min():.1f} p1 {np.percentile(nb,1):.1f} p5 {np.percentile(nb,5):.1f}  ratio min/maxblank {nb.min()/max(b.max(),1e-9):.2f}")
s=np.array([feats['mass'](i) for i in range(len(idx))])
o=np.argsort(s)
print('largest-blank pages by mass:',[(int(i),round(s[i],1)) for i in o[:260] if blank[i]][-8:])
print('smallest nonblank by mass:',[(int(i),round(s[i],1)) for i in o if not blank[i]][:10])
