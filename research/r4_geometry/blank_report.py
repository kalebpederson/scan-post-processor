import json,numpy as np,time,cv2
import geometry as G
idx=json.load(open('index.json'));lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
s=np.array(json.load(open('blank_scores.json')))
thr=G.BLANK_THRESHOLD
pred=s<thr
print('Operating point thr=%g'%thr)
print('confusion (rows truth, cols pred) over all 1969 pages [non-labelled 1643 pages assumed NOT_BLANK]:')
print(' BLANK     -> pred blank %d, pred not-blank %d'%((pred&blank).sum(),(~pred&blank).sum()))
print(' NOT_BLANK -> pred blank %d (FALSE BLANK), pred not-blank %d'%((pred&~blank).sum(),(~pred&~blank).sum()))
lb=np.array([str(i) in lab for i in range(len(idx))])
print('on the 326 labelled pages only: TP blank %d FN %d FP(false blank) %d TN %d'%((pred&blank&lb).sum(),(~pred&blank&lb).sum(),(pred&~blank&lb).sum(),(~pred&~blank&lb).sum()))
print('nearest NON-blank score %.1f -> margin thr/nearest = %.3f (nearest is %.1fx the threshold); max blank score %.1f'%(s[~blank].min(),thr/s[~blank].min(),s[~blank].min()/thr,s[blank].max()))
# repeated 50/50 splits: choose threshold = geometric mean(max train blank, min train nonblank) and test
rng=np.random.RandomState(0);fp=fn=0;ths=[]
for r in range(200):
    p=rng.permutation(len(idx));tr,te=p[:len(p)//2],p[len(p)//2:]
    t=np.sqrt(max(s[tr][blank[tr]].max(),1)*s[tr][~blank[tr]].min());ths.append(t)
    fp+=((s[te]<t)&~blank[te]).sum();fn+=((s[te]>=t)&blank[te]).sum()
print('200 random 50/50 splits, thr=geomean(train): thr mean %.1f sd %.1f; total held-out false-blank %d, missed-blank %d'%(np.mean(ths),np.std(ths),fp,fn))
# duplex stats
fr=[];bk=[];pg=[]
for i,m in enumerate(idx):
    if m['npages']%2==0: (fr if m['page']%2==0 else bk).append(pred[i])
print('blank overall %d/%d = %.1f%%'%(pred.sum(),len(pred),pred.mean()*100))
print('duplex PDFs (even page count): fronts blank %.1f%% (n=%d), backs blank %.1f%% (n=%d)'%(np.mean(fr)*100,len(fr),np.mean(bk)*100,len(bk)))
ks=sorted(set(m['k'] for m in idx));allblank=0;pdfs_with=0
for k in ks:
    ps=[i for i,m in enumerate(idx) if m['k']==k]
    if pred[ps].any(): pdfs_with+=1
print('PDFs containing >=1 blank page: %d of %d'%(pdfs_with,len(ks)))
# blank backs when front not blank
n=0;b=0
for k in ks:
    ps=[i for i,m in enumerate(idx) if m['k']==k]
    if len(ps)%2==0:
        for a,c in zip(ps[::2],ps[1::2]):
            if not pred[a]: n+=1;b+=pred[c]
print('back blank | front non-blank: %d/%d = %.1f%%'%(b,n,b/n*100))
# timing
ims=[cv2.imread(idx[i]['fn'],0) for i in range(0,1969,40)]
t=time.process_time();[G.blank_score(g,100) for g in ims];print('blank_score CPU %.0f ms/page at 100ppi'%((time.process_time()-t)/len(ims)*1000))
