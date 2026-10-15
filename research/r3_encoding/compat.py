"""Decoder compatibility: render candidate PDFs with poppler, mupdf, ghostscript, pdfium (+pdf.js via node) and compare; pikepdf open/check().
usage: compat.py cfg1,cfg2,... page1,page2,...  [dpi]"""
import sys, subprocess, itertools, warnings
from multiprocessing import Pool
from config import *
import render, metrics, pikepdf

def pdfjs(pdf, dpi):
    out = os.path.join(TMP, f"pj_{os.getpid()}_{os.path.basename(pdf)}.png")
    subprocess.run(["node", str(HERE / "pdfjs" / "render.mjs"), pdf, out, str(dpi)], check=True, capture_output=True, timeout=300)
    im = cv2.imread(out); os.remove(out); return im[:, :, ::-1].copy()
ENG = dict(render.ENGINES); ENG["pdfjs"] = pdfjs

def job(a):
    cfg, page, dpi = a
    pdf = str(WORK / "pdf" / cfg / f"{page}.pdf"); ref = load_clean(page)
    out = dict(cfg=cfg, page=page, engines={})
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        try:
            p = pikepdf.open(pdf); chk = p.check_pdf_syntax(); out["pikepdf_check"] = chk or "ok"; p.close()
        except Exception as e: out["pikepdf_check"] = "ERR " + str(e)
        out["pikepdf_warn"] = [str(w.message) for w in wl]
    R = {}
    for e, f in ENG.items():
        try: R[e] = metrics.fit(f(pdf, dpi), ref.shape) if dpi == 300 else f(pdf, dpi)
        except Exception as ex: out["engines"][e] = dict(error=str(ex)[:200])
    for e, r in R.items():
        out["engines"][e] = dict(shape=list(r.shape), ssim=metrics.fidelity(ref, r)["ssim"] if dpi == 300 else None)
    pairs = []
    for a_, b_ in itertools.combinations(R, 2):
        m, p999 = metrics.renderer_diff(R[a_], R[b_]); pairs.append((a_, b_, m, p999))
    out["pairs"] = pairs
    out["max_mean_diff"] = max((p[2] for p in pairs), default=None); out["max_p999"] = max((p[3] for p in pairs), default=None)
    return out

if __name__ == "__main__":
    cfgs = sys.argv[1].split(","); pgs = sys.argv[2].split(","); dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    J = [(c, p, dpi) for c in cfgs for p in pgs if (WORK / "pdf" / c / f"{p}.pdf").exists()]
    with Pool(int(os.environ.get("NPROC", 8))) as pool: res = pool.map(job, J, chunksize=1)
    json.dump(res, open(WORK / "res" / "compat.json", "w"), indent=1)
    for r in res:
        print(r["cfg"], r["page"], r["pikepdf_check"], "maxMAD %.2f p99.9 %.0f" % (r["max_mean_diff"] or -1, r["max_p999"] or -1), {e: (v.get("ssim") and round(v["ssim"], 3), v.get("error")) for e, v in r["engines"].items()})
