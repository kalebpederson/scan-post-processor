"""PDF page 1 -> RGB uint8 array using 4 independent engines."""
import os, subprocess, tempfile, uuid
import numpy as np, cv2
from config import TMP

def _tmp(ext): return os.path.join(TMP, uuid.uuid4().hex + ext)
def _read(p):
    im = cv2.imread(p, cv2.IMREAD_COLOR); os.remove(p); return im[:, :, ::-1].copy()

def poppler(pdf, dpi=300):
    base = _tmp("")
    subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", "-f", "1", "-l", "1", pdf, base], check=True, capture_output=True)
    return _read(base + ".png")
def mupdf(pdf, dpi=300):
    out = _tmp(".png")
    subprocess.run(["mutool", "draw", "-q", "-r", str(dpi), "-o", out, pdf, "1"], check=True, capture_output=True)
    return _read(out)
def ghostscript(pdf, dpi=300):
    out = _tmp(".png")
    subprocess.run(["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=png16m", f"-r{dpi}", "-dFirstPage=1", "-dLastPage=1", f"-sOutputFile={out}", pdf], check=True, capture_output=True)
    return _read(out)
def pdfium(pdf, dpi=300):
    import pypdfium2 as pdfium_
    d = pdfium_.PdfDocument(pdf)
    bm = d[0].render(scale=dpi / 72.0, rev_byteorder=True)      # RGB
    a = np.asarray(bm.to_numpy())[:, :, :3].copy(); d.close(); return a

ENGINES = dict(poppler=poppler, mupdf=mupdf, gs=ghostscript, pdfium=pdfium)
