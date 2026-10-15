import json,numpy as np
idx=json.load(open('index.json'));A=np.load('A6.npy');C=np.load('C6.npy')
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))]);nb=~blank
L=A[:,5]
others=A[:,[0,1,2,4]]
nag=(np.abs(others-L[:,None])<0.3).sum(1)
cl=C[:,5]
print('lsd conf pcts nonblank',np.round(np.percentile(cl[nb],[1,5,10,25,50]),2))
print('nag hist (nonblank)',np.bincount(nag[nb],minlength=5))
for t in (0.3,0.4,0.5,0.6):
    print('conf<',t,(cl[nb]<t).sum(),'| of which nag==0',((cl<t)&(nag==0)&nb).sum())
print('nag==0 & conf>=0.5:',((nag==0)&(cl>=0.5)&nb).sum(),' nag<=1 total',(nag[nb]<=1).sum())
# disagreement list
bad=np.where(nb&(nag==0))[0];print(len(bad))
json.dump([int(i) for i in bad],open('unrel_cand.json','w'))
for i in bad[:30]: print(i,idx[i]['k'],idx[i]['page'],np.round(A[i],2),np.round(C[i,[0,5]],2))
