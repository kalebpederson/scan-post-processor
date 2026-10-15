from config import *
from encoders import build_pdf
from configs import C
import pageselect as S, pikepdf
PG = ["pair02_front", "single13", "pair09_front", "pair07_front", "single09", "single26", "pair03_back", "single38", "pair06_front", "epson300"]
out = HERE / "samples"; out.mkdir(exist_ok=True)
def mk(name, fn):
    pgs = []; log = []
    for p in PG:
        pg, nm = fn(load_clean(p)); pgs.append(pg); log.append((p, nm, pg.payload()))
    f = str(out / name); build_pdf(pgs, f)
    pdf = pikepdf.open(f); pr = pdf.check_pdf_syntax(); print(name, os.path.getsize(f), len(pdf.pages), "pages; syntax problems:", pr); return log
for name, cfg in (("sample_R2_E3x_mrc_c100_q40_444.pdf", "E3x_c100_q40_444_mean"), ("sample_R2_E1_jpeg150_q55_444.pdf", "E1_j150_q55_444"), ("sample_R2_E1_jpeg200_q55_444.pdf", "E1_j200_q55_444"),
                  ("sample_R2_E5a_jpeg150_q40_444_blacktext.pdf", "E5a_q40_444_t3"), ("sample_R2_E5b_jpeg150_q40_444_blacktext_removed.pdf", "E5b_q40_444_t3")):
    mk(name, lambda r, c=cfg: (C[c]["fn"](r, **C[c]["kw"]), c))
log = mk("sample_R2_RECOMMENDED_per_page_selection.pdf", S.encode_page); print(log)
