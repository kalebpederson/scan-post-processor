"""Candidate registry: name -> (family, encoder fn, kwargs, applies_to) ; applies_to: 'all' or 'gray' (only pages classified no-colour)."""
from encoders import *
C = {}
def add(name, fam, fn, applies="all", **kw): C[name] = dict(fam=fam, fn=fn, kw=kw, applies=applies)

# ---- E1 JPEG
for r in (150, 200):
    for q in (40, 55, 70, 85):
        for sub, sn in ((2, "420"), (0, "444")):
            add(f"E1_j{r}_q{q}_{sn}", "E1", enc_jpeg, r=r, q=q, sub=sub)
        add(f"E1g_j{r}_q{q}", "E1g", enc_jpeg, "gray", r=r, q=q, sub=0, gray=True)
# ---- E2 palette
for r in (150, 200):
    for k in (4, 8, 16, 32):
        for m, mn in (("kmeans", "km"), ("mediancut", "mc")):
            add(f"E2_p{r}_k{k}_{mn}", "E2", enc_palette, r=r, k=k, method=m)
    for k in (4, 8, 16):
        add(f"E2g_p{r}_g{k}_km", "E2g", enc_palette, "gray", r=r, k=k, method="kmeans", gray=True)
        add(f"E2g_p{r}_g{k}_uni", "E2g", enc_palette, "gray", r=r, k=k, method="uniform", gray=True)
# ---- E3 mask + colour
BASE3 = dict(mask_ppi=300, codec="jbig2", col_ppi=100, q=55, sub=2, thr=ADAPT)
def m3(name, **ch):
    kw = dict(BASE3); kw.update(ch); add(name, "E3", enc_mrc, **kw)
m3("E3_base")                       # mask300 jbig2, colour 100ppi q55 4:2:0, thr 96
for c in (75, 150): m3(f"E3_c{c}", col_ppi=c)
for q in (40, 70): m3(f"E3_q{q}", q=q)
m3("E3_444", sub=0)
for t in (64, 128): m3(f"E3_t{t}", thr=t)
m3("E3_a50", thr=(0.5, 32, 110))
for mp in (200, 400): m3(f"E3_m{mp}", mask_ppi=mp)
m3("E3_g4", codec="g4")
m3("E3_c75_q40", col_ppi=75, q=40)
m3("E3_c75_q40_m200", col_ppi=75, q=40, mask_ppi=200)
m3("E3_c50_q40", col_ppi=50, q=40)
m3("E3_c75_q30", col_ppi=75, q=30)
m3("E3_c75_q40_hl", col_ppi=75, q=40, hl=True)
m3("E3_c75_q40_hl25", col_ppi=75, q=40, hl=True, hl_w=25)
m3("E3_hl", hl=True)
add("E3g_base", "E3g", enc_mrc, "gray", gray=True, **BASE3)
add("E3g_c75_q40", "E3g", enc_mrc, "gray", gray=True, **dict(BASE3, col_ppi=75, q=40))
# ---- E4 stencils
for K in (2, 4, 7): add(f"E4_s{K}", "E4", enc_stencils, K=K, thr=ADAPT)
add("E4_s7_t64", "E4", enc_stencils, K=7, thr=64)
add("E4_s7_g4", "E4", enc_stencils, K=7, thr=ADAPT, codec="g4")

# ---- Round 2: E3 refinements (colour layer ppi / dominant-colour cells / pencil), E5 hybrids
for cp in (100, 120, 150):
    for q in (40, 55):
        for cm in ("mean", "dom"):
            m3(f"E3r_c{cp}_q{q}_{cm}", col_ppi=cp, q=q, colmode=cm, hl=True, hl_w=25)
m3("E3r_c100_q55_dom_lo12", col_ppi=100, q=55, colmode="dom", hl=True, hl_w=25, thr=(0.35, 12, 88))
m3("E3r_c100_q55_dom_dil", col_ppi=100, q=55, colmode="dom", hl=True, hl_w=25, dil=1)
m3("E3r_c150_q40_dom_lo12", col_ppi=150, q=40, colmode="dom", hl=True, hl_w=25, thr=(0.35, 12, 88))
m3("E3r_c150_q40_dom_dil", col_ppi=150, q=40, colmode="dom", hl=True, hl_w=25, dil=1)
for v in ("a", "b"):
    for q in (40, 55): add(f"E5{v}_q{q}", "E5", enc_hybrid, r=150, q=q, variant=v)
    add(f"E5{v}_q40_P150", "E5", enc_hybrid, r=150, q=40, variant=v, P=150)
    add(f"E5{v}_q40_P190", "E5", enc_hybrid, r=150, q=40, variant=v, P=190)
add("E5a_r200_q55", "E5", enc_hybrid, r=200, q=55, variant="a")
add("E5b_r200_q55", "E5", enc_hybrid, r=200, q=55, variant="b")

# ---- Round 2b: 4:4:4 colour layers / bases (4:2:0 chroma subsampling was the main colour-identity failure)
for cp, q, cm in ((75, 55, "mean"), (100, 40, "mean"), (100, 40, "dom"), (100, 55, "dom"), (120, 40, "mean"), (150, 40, "mean"), (75, 40, "mean")):
    m3(f"E3x_c{cp}_q{q}_444_{cm}", col_ppi=cp, q=q, sub=0, colmode=cm, hl=True, hl_w=25)
for v in ("a", "b"):
    for q in (40, 55): add(f"E5{v}_q{q}_444", "E5", enc_hybrid, r=150, q=q, sub=0, variant=v)
    add(f"E5{v}_r200_q40_444", "E5", enc_hybrid, r=200, q=40, sub=0, variant=v)

# E5 with a higher (~50% coverage) text-edge threshold: stencil text weight matches the reference
for v in ("a", "b"):
    add(f"E5{v}_q40_444_t2", "E5", enc_hybrid, r=150, q=40, sub=0, variant=v, thr=(0.5, 60, 135))
    add(f"E5{v}_q40_444_t3", "E5", enc_hybrid, r=150, q=40, sub=0, variant=v, thr=(0.55, 80, 150))
