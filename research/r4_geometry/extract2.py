import json,subprocess,os,cv2
from extract import ROOT
from multiprocessing import Pool
import pikepdf
idx=json.load(open('index.json'))
have={(m['k'],m['page']) for m in idx}
d=json.load(open(f'{ROOT}/research/survey.json'))
sc=[x for x in d['pdfs'] if x['class']=='scan']
todo=[(k,p) for k,x in enumerate(sc) for p in range(x['pages']) if (k,p) not in have]
def w(a):
    k,p=a
    fn=f'cache/{k:03d}_{p:02d}'
    subprocess.run(['pdftoppm','-r','100','-f',str(p+1),'-l',str(p+1),'-png','-singlefile',f'{ROOT}/{sc[k]["path"]}',fn],check=True)
    im=cv2.imread(fn+'.png')
    return dict(fn=fn+'.png',w=0,h=0,dpi=100.0,nimg=0,sw=im.shape[1],sh=im.shape[0],pdf=sc[k]['path'],page=p,npages=sc[k]['pages'],k=k,rendered=True)
if __name__=='__main__':
    with Pool(16) as pl: r=pl.map(w,todo)
    idx+=r; idx.sort(key=lambda m:(m['k'],m['page']))
    json.dump(idx,open('index.json','w')); print(len(idx))
