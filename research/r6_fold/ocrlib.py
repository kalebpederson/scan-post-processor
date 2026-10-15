"""Tesseract helper for the R6 study: word boxes (native pixel coordinates) at a reduced OCR resolution, single thread."""
import os, subprocess, tempfile, time
import numpy as np, cv2
from common import SCR

def tess_words(img, dpi, ocr_dpi=150.0, psm=3, min_conf=-1.0):
    """img: uint8 gray or RGB at `dpi`. Returns (words, cpu_seconds): words = list of dict(text, conf, x0, y0, x1, y1) in `img` pixel coordinates."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = min(1.0, ocr_dpi / dpi)
    if s < 0.999:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    with tempfile.TemporaryDirectory(dir=SCR) as td:
        p = os.path.join(td, 'p.png')
        cv2.imwrite(p, g)
        env = dict(os.environ, OMP_THREAD_LIMIT='1')
        import resource
        ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        r = subprocess.run(['tesseract', p, 'stdout', '--psm', str(psm), '-l', 'eng', 'tsv'], capture_output=True, text=True, env=env)
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)
    words = []
    for ln in r.stdout.splitlines()[1:]:
        f = ln.split('\t')
        if len(f) < 12 or f[0] != '5':
            continue
        conf = float(f[10])
        txt = f[11].strip()
        if not txt or conf < min_conf:
            continue
        x, y, w, h = (int(v) for v in f[6:10])
        words.append(dict(text=txt, conf=conf, x0=x / s, y0=y / s, x1=(x + w) / s, y1=(y + h) / s))
    return words, cpu
