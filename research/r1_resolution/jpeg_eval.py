import io, json
from multiprocessing import Pool
from PIL import Image
import epson_eval as E
from lib import *
CFG=[('lossless',None,None),('q75_420',75,2),('q85_420',85,2),('q92_420',92,2),('q100_420',100,2),('q92_444',92,0),('q75_444',75,0)]
RS=[133,150,200,300]
def job(a):
    name,q,ss=a
    im=Image.open(f'{SCANS}/epson_300dpi.png').convert('RGB')
    nbytes=None
    if q:
        b=io.BytesIO(); im.save(b,'JPEG',quality=q,subsampling=ss); nbytes=b.tell(); b.seek(0); im=Image.open(b); im.load()
    rgb=np.asarray(im)
    return name,nbytes,rgb
def run(a):
    name,nbytes,rgb=job(a); out=[]
    ref_rgb=np.asarray(Image.open(f'{SCANS}/epson_300dpi.png').convert('RGB'))
    for r in RS:
        def prep(x):
            x=cv2.rotate(x,cv2.ROTATE_90_CLOCKWISE); x=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY) if False else cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
            return x
        g=prep(rgb); g0=prep(ref_rgb)
        d=downsample(g,300,r) if r!=300 else g; d0=downsample(g0,300,r) if r!=300 else g0
        mse=np.mean((d.astype(float)-d0.astype(float))**2); psnr=float('inf') if mse==0 else 10*np.log10(255**2/mse)
        up=upsample(d,(7016,5096))
        s=E.score(E.ocr(up,600,E.PSM),1.0)
        out.append(dict(cfg=name,bytes=nbytes,r=r,psnr_gray=psnr,**{k:s[k] for k in ['ms_recall','pos_recall','pos_light','pos_mid','pos_bold','cer','hits']}))
    return out
if __name__=='__main__':
    with Pool(7) as p: res=sum(p.map(run,CFG),[])
    json.dump(res,open(f'{HERE}/jpeg_results.json','w'))
    for x in res: print(x['cfg'],x['bytes'],x['r'],round(x['psnr_gray'],1),round(x['ms_recall'],3),round(x['pos_recall'],3),round(x['cer'],3))
