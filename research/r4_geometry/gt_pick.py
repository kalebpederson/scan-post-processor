import json,numpy as np,cv2,random
from sheet import sheet
A=np.load('A.npy');idx=json.load(open('index.json'));ink=json.load(open('ink_raw.json'))
f=np.array([x['f0.15'] for x in ink])
a=np.abs(A)
ok=f>0.02
strict=np.where(ok&(a[:,:4].max(1)<0.13)&(a[:,4]<0.2))[0]
loose=np.where(ok&((a<0.13).sum(1)>=3)&(a.max(1)<0.7))[0]
loose=np.setdiff1d(loose,strict)
print(len(strict),len(loose))
random.seed(11)
# avoid taking >2 pages from same pdf
def pick(pool,n,used):
    pool=list(pool);random.shuffle(pool);out=[]
    for i in pool:
        if used.get(idx[i]['k'],0)>=2: continue
        used[idx[i]['k']]=used.get(idx[i]['k'],0)+1;out.append(i)
        if len(out)==n:break
    return out
used={}
c=pick(strict,36,used)+pick(loose,24,used)
json.dump([int(x) for x in c],open('gt_candidates.json','w'))
S=set(strict)
sheet([(cv2.imread(idx[i]['fn']),f"{i} {'S' if i in S else 'L'} {idx[i]['dpi']:.0f}") for i in c[:30]],cols=6,tw=260,out='gt_c1.png')
sheet([(cv2.imread(idx[i]['fn']),f"{i} {'S' if i in S else 'L'} {idx[i]['dpi']:.0f}") for i in c[30:]],cols=6,tw=260,out='gt_c2.png')
