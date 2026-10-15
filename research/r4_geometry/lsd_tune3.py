import json,numpy as np,pickle,itertools
from methods import _peak_hist
A=np.load('A7.npy');idx=json.load(open('index.json'))
S=pickle.load(open('lsd_segs.pkl','rb'))
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
P,H=A[:,0],A[:,1]
pseudo=(np.abs(P-H)<0.15)&~blank
G=(P+H)/2
print('pseudo-GT pages',pseudo.sum())
def est(L,minlen,wp,sm,mode,maxang=12.0):
    dx=L[:,2]-L[:,0];dy=L[:,3]-L[:,1];ln=np.hypot(dx,dy);k=ln>=100*minlen
    dx,dy,ln=dx[k],dy[k],ln[k]
    ang=np.degrees(np.arctan2(-dy,dx));ang=(ang+90)%180-90
    if mode=='h': dev=ang; m=np.abs(ang)<=maxang
    else:
        dev=np.where(np.abs(ang)<=45,ang,ang-np.sign(ang)*90); m=np.abs(dev)<=maxang
    if m.sum()==0: return 0.0
    return _peak_hist(dev[m],ln[m]**wp,-maxang,maxang,0.1,sm)[0]
ids=np.where(pseudo)[0]
res=[]
for minlen,wp,sm,mode in itertools.product((0.2,0.3,0.5,0.8,1.2,1.8),(1,2,3),(0.3,0.5),('hv','h')):
    e=np.abs(np.array([est(S[i],minlen,wp,sm,mode) for i in ids])-G[ids])
    res.append(((e<0.2).mean(),(e<0.35).mean(),minlen,wp,sm,mode))
res.sort(reverse=True)
for r in res[:10]: print(np.round(r[:2],3),r[2:])
print('baseline (0.3,2,0.35->0.3,hv):',[r for r in res if r[2:]==(0.3,2,0.3,'hv')][0][:2])
