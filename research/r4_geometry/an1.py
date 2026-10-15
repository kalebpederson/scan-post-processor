import json,numpy as np
R=json.load(open('survey_raw.json'));idx=json.load(open('index.json'))
N=list(R[0].keys())
A=np.array([[r[n][0] for n in N] for r in R]);C=np.array([[r[n][1] for n in N] for r in R])
print(N)
for i,n in enumerate(N): print(n,'|a| pct 50/90/99/max',np.round(np.percentile(np.abs(A[:,i]),[50,90,99,100]),2),'meanconf',C[:,i].mean().round(2))
med=np.median(A,1)
for i,n in enumerate(N):
    d=np.abs(A[:,i]-med); print(n,'dev from median-of-5: p50 %.2f p90 %.2f p99 %.2f frac>0.3 %.3f'%(*np.percentile(d,[50,90,99]),(d>0.3).mean()))
print(np.round(np.corrcoef(A.T),2))
# pairwise agreement within .3
for i in range(5):
    print(N[i],[round(float((np.abs(A[:,i]-A[:,j])<0.3).mean()),2) for j in range(5)])
np.save('A.npy',A);np.save('C.npy',C)
