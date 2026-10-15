import json,numpy as np
idx=json.load(open('index.json'));R=json.load(open('survey_raw3.json'))
N=list(R[0]);print(N)
A=np.array([[r[n][0] for n in N] for r in R]);C=np.array([[r[n][1] for n in N] for r in R]);T=np.array([[r[n][2] for n in N] for r in R])
np.save('A7.npy',A);np.save('C7.npy',C)
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))]);nb=~blank
ji={n:j for j,n in enumerate(N)}
for n in N:
    a=np.abs(A[nb,ji[n]]);print(f"{n:8s} nonblank |a| p50 {np.percentile(a,50):.2f} p90 {np.percentile(a,90):.2f} p99 {np.percentile(a,99):.2f} max {a.max():.1f}")
print('pairwise within 0.3 (nonblank):')
for n in N: print(f"{n:8s}",[round(float((np.abs(A[nb,ji[n]]-A[nb,ji[m]])<0.3).mean()),2) for m in N])
L,P=A[:,ji['lsd']],A[:,ji['proj']]
d=np.abs(L-P);print('lsd vs proj agree<0.3 nonblank',(d[nb]<0.3).mean(),'<0.15',(d[nb]<0.15).mean())
print('conf pctiles lsd',np.round(np.percentile(C[nb,ji['lsd']],[1,5,10,50]),2),'proj',np.round(np.percentile(C[nb,ji['proj']],[1,5,10,50]),2))
