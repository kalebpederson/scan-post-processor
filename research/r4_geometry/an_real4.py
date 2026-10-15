import json,numpy as np
idx=json.load(open('index.json'));A=np.load('A7.npy');C=np.load('C7.npy')
N=['proj','hough','cc_rect','cc_mom','grad','lsd'];ji={n:j for j,n in enumerate(N)}
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))]);nb=~blank
L,P=A[:,5],A[:,0];d=np.abs(L-P)
dis=nb&(d>=0.3);print('disagree',dis.sum())
others=A[:,[1,2,3,4]]
supL=(np.abs(others-L[:,None])<0.3).sum(1);supP=(np.abs(others-P[:,None])<0.3).sum(1)
print('lsd better supported',(supL>supP)[dis].sum(),'proj better',(supP>supL)[dis].sum(),'tie',(supP==supL)[dis].sum())
for lo,hi in ((0.3,0.6),(0.6,1),(1,3),(3,30)):
    m=dis&(d>=lo)&(d<hi);print(lo,hi,m.sum(),'lsd sup',(supL>supP)[m].sum(),'proj sup',(supP>supL)[m].sum())
json.dump([int(i) for i in np.where(dis&(d>=0.6))[0]],open('dis_big.json','w'))
print('proj at exactly 0.0:',(A[nb,0]==0).mean(),' proj conf for those',np.mean(C[nb&(A[:,0]==0),0]))
print('proj at |a|>=11.9 (bounds):',(np.abs(A[nb,0])>=11.9).sum())
