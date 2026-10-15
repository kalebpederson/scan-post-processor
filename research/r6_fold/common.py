"""Shared helpers for the R6 fold study (paths, loaders)."""
import json, os, sys
import numpy as np, cv2
cv2.setNumThreads(int(os.environ.get('R6_CV_THREADS', '1')))     # the studies run 12-14 worker processes: one OpenCV thread each
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCR = os.path.join(ROOT, 'research', 'r6_fold', 'work')
sys.path.insert(0, ROOT + '/src')
CACHE = ROOT + '/research/r4_geometry/cache/'
_idx = None

def index():
    global _idx
    if _idx is None:
        _idx = json.load(open(ROOT + '/research/r4_geometry/index.json'))
    return _idx

def corpus_page(i):
    """i-th entry of the R4 index -> RGB uint8 at ~100 ppi (cache) and its record."""
    r = index()[i]
    return cv2.imread(ROOT + '/research/r4_geometry/' + r['fn'])[:, :, ::-1], r

def td_page(p, dpi_out=None):
    """Test-document page p (1-based) at 300 ppi (scratch .npy, row-padded p8) or area-downscaled to dpi_out."""
    a = np.load(f'{SCR}/td/p{p:02d}.npy')
    if dpi_out and dpi_out != 300:
        s = dpi_out / 300.0
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return a

def native_page(i):
    """Full-resolution RGB of the i-th R4 index page, straight from the (read-only) source PDF."""
    import pikepdf
    from pikepdf import PdfImage
    r = index()[i]
    pdf = pikepdf.open(ROOT + '/' + r['pdf'])
    page = pdf.pages[r['page']]
    imgs = list(page.images.items())
    name, raw = max(imgs, key=lambda kv: int(kv[1].Width) * int(kv[1].Height))
    return np.asarray(PdfImage(raw).as_pil_image().convert('RGB')), r
