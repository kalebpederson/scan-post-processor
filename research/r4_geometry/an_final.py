import json,numpy as np
idx=json.load(open('index.json'));R=json.load(open('final_survey.json'))
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))]);nb=~blank
ang=np.array([r['ang'] for r in R]);conf=np.array([r['conf'] for r in R])
print('CPU s/page median %.3f p90 %.3f'%(np.median([r['cpu_s'] for r in R]),np.percentile([r['cpu_s'] for r in R],90)))
print('nonblank pages',nb.sum(),'conf hist',np.bincount((conf[nb]*3).round().astype(int),minlength=4),' (conf 0,1/3,2/3,1)')
rel=nb&(conf>=0.6);print('reliable',rel.sum(),'%.1f%%'%(rel.sum()/nb.sum()*100),'unreliable',(nb&~rel).sum())
a=np.abs(ang[rel])
print('reliable |angle| p50 %.2f p75 %.2f p90 %.2f p95 %.2f p99 %.2f max %.2f'%(*np.percentile(a,[50,75,90,95,99]),a.max()))
for t in (0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2):
    print(f'  |a|>{t}: {(a>t).sum()} ({(a>t).mean()*100:.1f}% of reliable)')
# histogram signed
h,e=np.histogram(ang[rel],bins=np.arange(-3,3.01,0.25))
for x,y in zip(e[:-1],h): print('%+.2f %4d %s'%(x,y,'#'*int(y/10)))
# fronts/backs
fr=[];bk=[];pairs=[]
for i,m in enumerate(idx):
    if not rel[i] or m['npages']%2: continue
    (fr if m['page']%2==0 else bk).append(ang[i])
for nm,x in (('front',np.array(fr)),('back',np.array(bk))):
    print(nm,len(x),'mean signed %+.3f median %+.3f sd %.3f |a| p50 %.2f p90 %.2f p99 %.2f max %.2f frac>0.25 %.3f'%(x.mean(),np.median(x),x.std(),*np.percentile(np.abs(x),[50,90,99,100]),(np.abs(x)>0.25).mean()))
from scipy import stats
print('Mann-Whitney front vs back signed',stats.mannwhitneyu(fr,bk).pvalue,' |a|',stats.mannwhitneyu(np.abs(fr),np.abs(bk)).pvalue)
# pairs
P=[]
for k in set(m['k'] for m in idx):
    ps=[i for i,m in enumerate(idx) if m['k']==k]
    if len(ps)%2==0:
        for a_,b_ in zip(ps[::2],ps[1::2]):
            if rel[a_] and rel[b_]: P.append((ang[a_],ang[b_]))
P=np.array(P);print('sheets (front,back) both reliable',len(P),'corr %.2f; mean(back-front) %.3f; corr(front,back) after removing means ...'%(np.corrcoef(P.T)[0,1],(P[:,1]-P[:,0]).mean()))
# by dpi group
for d in (150,200,300):
    k=np.array([abs(m['dpi']-d)<10 for m in idx])&rel
    print('dpi',d,k.sum(),'|a| p50 %.2f p90 %.2f'%tuple(np.percentile(np.abs(ang[k]),[50,90])))
# unreliable list
un=np.where(nb&(conf<0.6))[0];print('unreliable',len(un),[ (int(i),idx[i]['k'],idx[i]['page']) for i in un[:10]])
json.dump({'unrel':[int(i) for i in un]},open('unrel.json','w'))
