"""OCR legibility on PDFium (Chrome) renders for selected configs: epson300 + native-300 corpus pages. usage: run_pdfium_ocr.py cfgglob"""
import sys, fnmatch
from multiprocessing import Pool
from config import *
import render, ocrlib
OUT = WORK / "res" / "pdfium_ocr"
NAT = {p["name"] for p in ocrlib.native300_pages()} | {"epson300"}
def job(a):
    c, p = a; o = OUT / c / f"{p}.json"
    if o.exists(): return
    r = render.pdfium(str(WORK / "pdf" / c / f"{p}.pdf"))
    res = ocrlib.epson_eval(r) if p == "epson300" else ocrlib.corpus_eval(p, r)
    o.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(o, "w"))
if __name__ == "__main__":
    pats = sys.argv[1].split(",")
    cfgs = sorted(d.name for d in (WORK / "pdf").iterdir() if any(fnmatch.fnmatch(d.name, q) for q in pats))
    J = [(c, p) for c in cfgs for p in sorted(NAT) if (WORK / "pdf" / c / f"{p}.pdf").exists()]
    J.sort(key=lambda j: j[1] != "epson300")
    with Pool(16) as pool:
        for _ in pool.imap_unordered(job, J, chunksize=1): pass
