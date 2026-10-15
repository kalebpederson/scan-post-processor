import json,numpy as np
idx=json.load(open('index.json'))
old=np.load('A.npy');oldC=np.load('C.npy')   # proj,hough,cc_rect,cc_mom,grad(old)
R2=json.load(open('survey_raw2.json'))
N=['proj','hough','cc_rect','cc_mom','grad','lsd']
A=np.zeros((len(idx),6));C=np.zeros((len(idx),6))
A[:,0]=[r['proj'][0] for r in R2];C[:,0]=[r['proj'][1] for r in R2]
A[:,1]=old[:,1];C[:,1]=oldC[:,1];A[:,2]=old[:,2];C[:,2]=oldC[:,2];A[:,3]=old[:,3];C[:,3]=oldC[:,3]
A[:,4]=[r['grad'][0] for r in R2];C[:,4]=[r['grad'][1] for r in R2];A[:,5]=[r['lsd'][0] for r in R2];C[:,5]=[r['lsd'][1] for r in R2]
np.save('A6.npy',A);np.save('C6.npy',C)
lab=json.load(open('labels_blank.json'))['labels']
ink=json.load(open('ink_raw.json'));fc=np.array([x['f0.1'] for x in ink])
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
nb=~blank
print('blank',blank.sum(),'nonblank',nb.sum())
for j,n in enumerate(N):
    a=np.abs(A[nb,j]);print(n,'|a| nonblank p50/p90/p99/max',np.round(np.percentile(a,[50,90,99,100]),2))
# agreement of each method with lsd
for j,n in enumerate(N):
    d=np.abs(A[nb,j]-A[nb,5]);print(n,'vs lsd: within .15 %.2f within .3 %.2f within .5 %.2f'%tuple((d<t).mean() for t in (.15,.3,.5)))
# lsd histogram
h,e=np.histogram(A[nb,5],bins=np.arange(-2.5,2.6,0.25));
for a,b in zip(e[:-1],h): print('%+.2f %4d %s'%(a,b,'#'*int(b/12)))
# fronts vs backs: pdfs with even number of pages >=4, 
fr=[];bk=[]
for i,m in enumerate(idx):
    if not nb[i]: continue
    if m['npages']%2==0 and m['npages']>=2:
        (fr if m['page']%2==0 else bk).append(A[i,5])
fr=np.array(fr);bk=np.array(bk)
for nm,x in (('front(odd pp)',fr),('back(even pp)',bk)):
    print(nm,len(x),'mean signed %.3f median %.3f |a| p50 %.2f p90 %.2f p99 %.2f max %.2f frac|a|>0.3 %.3f'%(x.mean(),np.median(x),*np.percentile(np.abs(x),[50,90,99,100]),(np.abs(x)>0.3).mean()))
# paired: front vs back within sheet - correlation
pairs=[]
for k in set(m['k'] for m in idx):
    ps=[i for i,m in enumerate(idx) if m['k']==k]
    if len(ps)%2==0:
        for a,b in zip(ps[::2],ps[1::2]):
            if nb[a] and nb[b]: pairs.append((A[a,5],A[b,5]))
pairs=np.array(pairs);print('pairs',len(pairs),'corr signed %.2f; corr(front,-back) etc'%np.corrcoef(pairs.T)[0,1])
