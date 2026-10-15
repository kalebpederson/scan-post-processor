"""Round-2 metrics over (cfg, page) PDFs, rendered with PDFium at the reference grid. usage: run_metrics2.py cfgpattern[,cfgpattern...]"""
import sys, fnmatch
from multiprocessing import Pool
from config import *
import metrics2
import pypdfium2 as pdfium_
OUT = WORK / "res" / "m2"
def render_ref_grid(pdf, shape):
    d = pdfium_.PdfDocument(pdf); pg = d[0]; wpt = pg.get_width(); sc = shape[1] / wpt
    a = np.asarray(pg.render(scale=sc, rev_byteorder=True).to_numpy())[:, :, :3].copy(); d.close(); return a
def job(a):
    cfg, name = a; o = OUT / cfg / f"{name}.json"
    if o.exists(): return
    ref = load_clean(name); info = metrics2.RefInfo(name, ref)
    r = metrics2.evaluate(info, render_ref_grid(str(WORK / "pdf" / cfg / f"{name}.pdf"), ref.shape))
    o.parent.mkdir(parents=True, exist_ok=True); json.dump(r, open(o, "w"))
if __name__ == "__main__":
    pats = sys.argv[1].split(","); pnames = [r["name"] for r in json.load(open(WORK / "res" / "pages.json"))]
    cfgs = sorted(d.name for d in (WORK / "pdf").iterdir() if any(fnmatch.fnmatch(d.name, p) for p in pats))
    J = [(c, p) for c in cfgs for p in pnames if (WORK / "pdf" / c / f"{p}.pdf").exists() and not c.endswith("_orig")]
    print(len(J), "jobs", flush=True)
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for i, _ in enumerate(pool.imap_unordered(job, J, chunksize=2)):
            if i % 200 == 0: print(i, flush=True)
