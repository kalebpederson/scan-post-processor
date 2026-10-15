"""Step 2: run every (config, page): encode -> single-page PDF -> poppler render @300 -> fidelity (+pdfium fidelity) -> OCR legibility. Resumable."""
import sys, time, fnmatch, traceback
from multiprocessing import Pool
from config import *
from configs import C
from encoders import build_pdf, overhead_bytes
import render, metrics, ocrlib

RES = WORK / "res"
PAGES = {p["name"]: p for p in pages()}
INFO = {r["name"]: r for r in json.load(open(RES / "pages.json"))}
OVERHEAD = overhead_bytes()
NATIVE300 = {p["name"] for p in ocrlib.native300_pages()}

def job(a):
    cname, pname = a
    out = RES / "cfg" / cname / f"{pname}.json"
    if out.exists(): return None
    cfg = C[cname]; info = INFO[pname]
    try:
        rgb = load_clean(pname)
        t = time.perf_counter(); pg = cfg["fn"](rgb, **cfg["kw"]); te = time.perf_counter() - t
        pdfp = WORK / "pdf" / cname; pdfp.mkdir(exist_ok=True)
        pdf = str(pdfp / f"{pname}.pdf")
        t = time.perf_counter(); build_pdf([pg], pdf); tb = time.perf_counter() - t
        size = os.path.getsize(pdf)
        rp = render.poppler(pdf)
        res = dict(cfg=cname, page=pname, cls=info["cls"], bytes=size - OVERHEAD, payload=pg.payload(), t_enc=te, t_build=tb,
                   fid=metrics.fidelity(rgb, rp))
        if pname == "epson300": res["ocr"] = ocrlib.epson_eval(rp)
        elif pname in NATIVE300: res["ocr"] = ocrlib.corpus_eval(pname, rp)
        out.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"))
        return cname, pname
    except Exception:
        sys.stderr.write(f"FAIL {cname} {pname}\n{traceback.format_exc()}\n"); return None

def jobs(pattern):
    J = []
    for cname, cfg in C.items():
        if not fnmatch.fnmatch(cname, pattern): continue
        for pname, info in INFO.items():
            if cfg["applies"] == "gray" and info["cls"] != "gray": continue
            J.append((cname, pname))
    return J

if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*"      # config glob
    pagef = sys.argv[2] if len(sys.argv) > 2 else "*"
    J = [j for j in jobs(pattern) if any(fnmatch.fnmatch(j[1], q) for q in pagef.split(","))]
    # epson first (longest OCR), then the rest
    J.sort(key=lambda j: (j[1] != "epson300", j[1] not in NATIVE300))
    if not ocrlib.CORPUS_REF_JSON.exists(): ocrlib.build_corpus_ref()
    print(len(J), "jobs", flush=True)
    t0 = time.time(); n = 0
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for r in pool.imap_unordered(job, J, chunksize=1):
            n += 1
            if n % 50 == 0: print(n, "/", len(J), "%.0fs" % (time.time() - t0), flush=True)
