"""Find pages with non-white paper: modal paper colour of the page (via ref_background, interior only)."""
import sys, json; sys.path.insert(0,'.')
from common import *
from multiprocessing import Pool
def f(it):
    name,path,dpi=it
    rgb=load_rgb(path); bg=ref_background(rgb,dpi)
    h,w=bg.shape[:2]; c=bg[h//10:-h//10, w//8:-w//10].reshape(-1,3)
    med=np.median(c,0); 
    # lab chroma of paper
    lab=cv2.cvtColor(np.uint8([[med]]),cv2.COLOR_RGB2Lab)[0,0].astype(float)
    chroma=float(np.hypot(lab[1]-128,lab[2]-128))
    # illumination non-uniformity: p5/p95 of bg luminance
    l=lum(c); return dict(name=name,dpi=dpi,paper=[round(float(x)) for x in med],paper_chroma=round(chroma,1),lum_p5=round(float(np.percentile(l,5))),lum_p95=round(float(np.percentile(l,95))))
if __name__=="__main__":
    its=corpus_items()
    with Pool(16) as p: r=p.map(f,its)
    json.dump(r,open('paper_survey.json','w'),indent=1)
    for x in sorted(r,key=lambda x:-x['paper_chroma']): print(x)
