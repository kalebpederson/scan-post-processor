import sys, pickle, json; sys.path.insert(0,'.')
import numpy as np
from run_eval import COLORED_PAPER
cfgs, agg, res = pickle.load(open('work/final.pkl','rb'))
def hue(c):
    pc=[r[c] for n,r in res if not n.startswith('bl:') and 'color' in r[c] and n not in COLORED_PAPER and n!='epson']
    dh=np.concatenate([p['color']['dh'] for p in pc]); cr=np.concatenate([p['color']['cr'] for p in pc])
    return dh.mean(), np.percentile(dh,90), np.median(cr), np.percentile(cr,10)
names=[c[0] for c in cfgs]
L=[]
L.append("| metric | "+" | ".join(names)+" |"); L.append("|---|"+"---|"*len(names))
def row(label,f): L.append("| "+label+" | "+" | ".join(f(agg[n],n) for n in names)+" |")
row("paper whiteness, 53 white-paper pages (mean / worst page)",lambda a,n:f"{a['white_whitepaper']:.4f} / {a['white_whitepaper_min']:.4f}")
row("background-clean fraction (non-ink area >=250)",lambda a,n:f"{a['bk_clean']:.3f}")
for p in ("single08","single38","image01"):
    row(f"colored paper {p}: whiteness (flatten)",lambda a,n,p=p:f"{a['white_colored_pages'][p]:.3f}")
row("ink median / p90 luminance (0=black)",lambda a,n:f"{a['ink_med']} / {a['ink_p90']}")
row("ink-area ratio (dark px out/in)",lambda a,n:f"{a.get('ink_area_ratio',float('nan')):.2f}")
row("hue shift mean / p90 (deg)",lambda a,n:"%.1f / %.1f"%hue(n)[:2])
row("chroma ratio median / p10",lambda a,n:"%.2f / %.2f"%(hue(n)[2],hue(n)[3]))
row("faint-content retention (17 regions, core px)",lambda a,n:f"{a['pos_ret']:.3f}")
for cl in ("pencil","grid","light_print","ruled","highlighter","light_pen"):
    row("  - "+cl,lambda a,n,cl=cl:f"{a['pos_by_class'][cl]:.2f}")
row("faint contrast: out dE / native dE",lambda a,n:f"{a['pos_out_dE']:.1f} / {a['pos_nat_dE']:.1f}")
row("bleed-only regions survival (11 regions)",lambda a,n:f"{a['neg_surv_regions']:.3f}")
row("pair bleed survival, pixel-pooled (35 directed)",lambda a,n:f"{a['bleed_surv_pooled']:.3f}")
row("pair bleed survival, per-instance mean",lambda a,n:f"{a['bleed_surv_macro']:.3f}")
row("mean darkness in bleed region (native visible-bleed darkness 0.0515)",lambda a,n:f"{a['bleed_dark_vis']:.4f}")
open('work/tables.md','w').write("\n".join(L)); print("\n".join(L))
print(); 
for n in names: print(n,'pos_each',agg[n]['pos_each'])
for n in names[:2]+names[1:2]: print(n,'neg_each',agg[n]['neg_each'])
