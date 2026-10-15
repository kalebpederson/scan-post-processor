import json, collections
from multiprocessing import Pool
from skimage.metrics import structural_similarity as ssim
from lib import *
PSM=3
man=json.load(open(f'{CORPUS}/manifest.json'))
FILES=[m['file'] for m in man if m['dpi'][0]==300]
def words(g,dpi,minconf=None):
    w=ocr(g,dpi,PSM)
    return [x for x in w if minconf is None or x['conf']>=minconf]
def recall(ref,hyp):
    rc=collections.Counter(w['text'] for w in ref); hc=collections.Counter(w['text'] for w in hyp)
    return sum(min(rc[t],hc[t]) for t in rc)/max(1,len(ref))
def job(f):
    g=cv2.imread(f'{CORPUS}/{f}',cv2.IMREAD_GRAYSCALE)
    ref=words(g,300,90); res=dict(file=f,nref=len(ref),rows=[])
    h=np.array([w['h'] for w in ref]) if ref else np.array([0])
    res['median_word_h_px']=float(np.median(h)); res['p10_word_h_px']=float(np.percentile(h,10))
    for r in RATES:
        if r==300: im=g; td=0
        else:
            t=time.perf_counter(); im=downsample(g,300,r); td=(time.perf_counter()-t)*1e3
        up=upsample(im,(g.shape[1],g.shape[0])) if r!=300 else g
        rt=recall(ref,words(up,300)); di=recall(ref,words(im,r)) if r!=300 else rt
        res['rows'].append(dict(r=r,rt=rt,direct=di,ssim=float(ssim(g,up,data_range=255)) if r!=300 else 1.0,t_down_ms=td))
    return res
if __name__=='__main__':
    with Pool(14) as p: out=p.map(job,FILES,chunksize=1)
    json.dump(out,open(f'{HERE}/corpus_results.json','w'),indent=1)
    for o in out: print(o['file'],o['nref'],[round(x['rt'],2) for x in o['rows']])
