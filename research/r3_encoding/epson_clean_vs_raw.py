"""Epson keymap: same encoder on UNCLEANED vs CLEANED 300 ppi input; OCR legibility of the decoded PDF (poppler and pdfium renders)."""
from multiprocessing import Pool
from config import *
import render, ocrlib
from encoders import build_pdf
from configs import C
CFGS = ["E1_j150_q55_420", "E1_j200_q70_420", "E1_j200_q85_420", "E2_p200_k16_km", "E3_c75_q40", "E3_base", "E4_s7", "E4_s2"]
def job(a):
    cfg, which = a
    p = [q for q in pages() if q["name"] == "epson300"][0]
    rgb = load_clean("epson300") if which == "clean" else load_src(p)
    pg = C[cfg]["fn"](rgb, **C[cfg]["kw"]); pdf = str(WORK / "tmp" / f"ev_{cfg}_{which}.pdf"); build_pdf([pg], pdf)
    out = dict(cfg=cfg, input=which, bytes=os.path.getsize(pdf))
    for e in ("poppler", "pdfium"):
        s = ocrlib.epson_eval(render.ENGINES[e](pdf)); out[e] = {k: s[k] for k in ("pos_recall", "pos_light", "pos_mid", "pos_bold", "ms_recall")}
    return out
if __name__ == "__main__":
    J = [(c, w) for c in CFGS for w in (("raw", "clean") if c.startswith(("E1", "E2")) else ("clean",))]  # mask encoders need flattened paper, so raw only for E1/E2
    with Pool(16) as pool: res = pool.map(job, J, chunksize=1)
    json.dump(res, open(WORK / "res" / "epson_clean_vs_raw.json", "w"), indent=1)
    for r in res: print(r["cfg"], r["input"], r["bytes"], "poppler pos %.3f light %.3f | pdfium pos %.3f light %.3f" % (r["poppler"]["pos_recall"], r["poppler"]["pos_light"], r["pdfium"]["pos_recall"], r["pdfium"]["pos_light"]))
