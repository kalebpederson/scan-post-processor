import sys, pickle, re; sys.path.insert(0,'.')
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
files = sys.argv[2:]; out = sys.argv[1]
R = {}
for f in files:
    o = pickle.load(open(f, 'rb')); R.update(o[1])
def fam(n):
    if n == "B0": return "B0 (global levels)"
    if n.startswith("FINAL"): return "FINAL (chosen)"
    if n.startswith("M1"): return "M1 flat-field + levels (w sweep)"
    if n.startswith("M2_") and "_g1.0" in n: return "M2 soft threshold, no boost"
    if n.startswith("M2_"): return "M2 soft threshold + gamma boost"
    if n.startswith("M3s"): return "M3 sharpness gate"
    if n.startswith("M2h"): return "hysteresis (early)"
    if n.startswith("L7"): return "hysteresis + line support"
    if n.startswith("HS"): return "hysteresis + sharpness bonus"
    if n.startswith("H"): return "hysteresis (+faint gain)"
    return "other"
col = {}; cm = plt.get_cmap("tab10")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
for ax, (xk, xl) in zip(axs, (("neg_surv_regions", "bleed-only-region survival (mean over 11 regions)"), ("bleed_surv_pooled", "pair bleed survival (pixel-pooled, 35 sheets-directions)"))):
    for n, a in R.items():
        f = fam(n); col.setdefault(f, cm(len(col) % 10))
        if n in ("keep", "FINAL_keep", "M2_threshold_matched", "M1_flatfield_levels", "M2_threshold_retain"): continue
        ax.scatter(a[xk], a["pos_ret"], color=col[f], s=140 if n == "B0" else 28, marker="*" if n == "B0" else "o", label=f, zorder=3)
        if n == "FINAL_flatten": ax.scatter(a[xk], a["pos_ret"], s=260, facecolors="none", edgecolors="red", linewidths=2, zorder=4); ax.annotate("chosen", (a[xk], a["pos_ret"]), xytext=(8, -16), textcoords="offset points", color="red")
    h, l = ax.get_legend_handles_labels(); u = dict(zip(l, h)); ax.legend(u.values(), u.keys(), fontsize=7, loc="lower right")
    pts = sorted([(a[xk], a['pos_ret']) for a in R.values()]); fr = []; best = -1
    for x, y in pts:
        if y > best: fr.append((x, y)); best = y
    ax.plot(*zip(*fr), color='k', lw=1, alpha=.5, label='Pareto frontier')
    ax.axvline(0.02, color="gray", ls=":"); ax.set_xlabel(xl + "  (lower = better)"); ax.set_ylabel("faint real content retention (higher = better)"); ax.grid(alpha=.3)
    ax.set_xlim(-0.02, 0.9); ax.set_ylim(0.45, 0.95)
plt.suptitle("Bleed-through suppression vs faint-content retention (stroke-core metric)"); plt.tight_layout(); plt.savefig(out, dpi=130)
