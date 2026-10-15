import numpy as np, pikepdf, json, os
from pikepdf import PdfImage
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r4_geometry/work'; _os.makedirs(_WORK, exist_ok=True)
ROOT=_ROOT
def get_full(m):
    """full-resolution RGB for an index.json entry"""
    import cv2
    if m.get('rendered'):
        import subprocess
        fn=_WORK + '/r_%d_%d'%(m['k'],m['page'])
        subprocess.run(['pdftoppm','-r','200','-f',str(m['page']+1),'-l',str(m['page']+1),'-png','-singlefile',f"{ROOT}/{m['pdf']}",fn],check=True)
        return cv2.imread(fn+'.png')[:,:,::-1]
    pdf=pikepdf.open(os.path.join(ROOT,m['pdf']))
    page=pdf.pages[m['page']]
    name,raw=max(page.images.items(),key=lambda kv:int(kv[1].Width)*int(kv[1].Height))
    return np.asarray(PdfImage(raw).as_pil_image().convert('RGB'))
