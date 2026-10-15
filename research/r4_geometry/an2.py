import json,numpy as np
A=np.load('A.npy');idx=json.load(open('index.json'))
med=np.median(A,1)
o=np.argsort(-np.abs(med))[:40]
for i in o: print(i,idx[i]['k'],idx[i]['page'],round(med[i],2),np.round(A[i],2),idx[i]['pdf'][-50:])
print((np.abs(med)>0.5).sum(),(np.abs(med)>1).sum(),(np.abs(med)>0.3).sum())
