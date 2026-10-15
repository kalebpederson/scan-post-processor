"""E0 = the user's current pipeline, run on the original uncleaned pages (as a PDF per page). Variants:
 E0         : literal command + '-compress jpeg' (IM 7.1.2 defaults to RunLength for PDF output; the user's ~468 KB/page files are JPEG, so jpeg is assumed)
 E0_noalpha : same + '-alpha off' (drops the 8-bit SMask that magick adds when rasterising the PDF)
 E0_literal : the command exactly as given (size/time only)
Output page is 220x285 pt (=2544x3295 px at ~833 ppi): -density 200 -page Letter does NOT resample, it only relabels."""
import subprocess, time, sys
from multiprocessing import Pool
from config import *
import e0, render, metrics, ocrlib
from encoders import overhead_bytes
RES = WORK / "res"
INFO = {r["name"]: r for r in json.load(open(RES / "pages.json"))}
PAGES = {p["name"]: p for p in pages()}
NATIVE300 = {p["name"] for p in ocrlib.native300_pages()}
VARIANTS = {"E0": ["-compress", "jpeg"], "E0_noalpha": ["-compress", "jpeg", "-alpha", "off"], "E0_literal": []}

def poppler_to(pdf, w, h):
    base = os.path.join(TMP, "e0r_" + os.path.basename(pdf)[:-4] + "_" + str(os.getpid()))
    subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", str(w), "-scale-to-y", str(h), "-f", "1", "-l", "1", pdf, base], check=True, capture_output=True)
    im = cv2.imread(base + ".png"); os.remove(base + ".png"); return im[:, :, ::-1].copy()

def job(a):
    var, pname = a
    out = RES / "cfg" / var / f"{pname}.json"
    if out.exists(): return
    p = PAGES[pname]; d = WORK / "pdf" / var; d.mkdir(exist_ok=True, parents=True)
    inp = str(d / (pname + "_orig.pdf")); outp = str(d / (pname + ".pdf"))
    e0.original_pdf(p, inp)
    cmd = list(e0.CMD); cmd[3] = inp; cmd[-1:] = VARIANTS[var] + [outp]
    env = dict(os.environ, MAGICK_THREAD_LIMIT="1", OMP_NUM_THREADS="1")
    t = time.perf_counter(); subprocess.run(cmd, check=True, capture_output=True, env=env); dt = time.perf_counter() - t
    os.remove(inp)
    res = dict(cfg=var, page=pname, cls=INFO[pname]["cls"], bytes=os.path.getsize(outp) - 600, payload=os.path.getsize(outp), t_enc=dt)
    if var != "E0_literal":
        ref = load_clean(pname); h, w = ref.shape[:2]
        r = poppler_to(outp, w, h)
        res["fid"] = metrics.fidelity(ref, r)
        if pname == "epson300": res["ocr"] = ocrlib.epson_eval(r)
        elif pname in NATIVE300: res["ocr"] = ocrlib.corpus_eval(pname, r)
    out.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"))

if __name__ == "__main__":
    J = [(v, n) for v in VARIANTS for n in INFO]
    J.sort(key=lambda j: j[1] != "epson300")
    with Pool(int(os.environ.get("NPROC", 16))) as pool:
        for _ in pool.imap_unordered(job, J, chunksize=1): pass
    print("e0 done")
