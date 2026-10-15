"""Builds results tables (work/<tag>/res/tables.md) and the Pareto plots (pareto_*.png)."""
import glob, collections
from config import *
import aggregate as A
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
RES = WORK / "res"
rows = A.load(); S = A.summarize(rows)
pd = collections.defaultdict(dict)
for f in glob.glob(str(RES / "pdfium_ocr" / "*" / "*.json")):
    c, p = f.split("/")[-2], f.split("/")[-1][:-5]; pd[c][p] = json.load(open(f))
import ocrlib
refn = {n: len(ocrlib.corpus_ref(n)) for n in [p["name"] for p in ocrlib.native300_pages()]}
for c, d in pd.items():
    r = [v["recall"] for p, v in d.items() if p != "epson300" and refn.get(p, 0) >= 20]
    S[c]["rec300_pdfium"] = sum(r) / len(r) if r else float("nan")
    if "epson300" in d: S[c].update(ep_pos_pdfium=d["epson300"]["pos_recall"], ep_light_pdfium=d["epson300"]["pos_light"])
comp = json.load(open(RES / "compat.json")); cm = {}
for r in comp:
    x = cm.setdefault(r["cfg"], dict(mad=0, p999=0, pj=0, minssim=1, bad=0, chk="ok"))
    for a, b, m, p in r["pairs"]:
        if "pdfjs" in (a, b): x["pj"] = max(x["pj"], m)
        else: x["mad"] = max(x["mad"], m); x["p999"] = max(x["p999"], p)
    x["minssim"] = min([x["minssim"]] + [v["ssim"] for v in r["engines"].values() if v.get("ssim")])
    if r["pikepdf_check"] != "ok" and r["pikepdf_check"] != []: x["chk"] = str(r["pikepdf_check"])[:40]
    x["bad"] += sum(1 for v in r["engines"].values() if "error" in v)
tm = json.load(open(RES / "timing.json"))
def row(c):
    s = S[c]; k = cm.get(c)
    return ("| %s | %.0f | %.0f | %.0f | %.1f | %.3f | %.3f | %s | %s | %.4f | %.1f | %.1f | %.3f | %.3f | %.4f | %s | %s |" % (
        c, s["median"] / 1e3, s["mean"] / 1e3, s["p90"] / 1e3, s["doc124"] / 1e6,
        s.get("rec300", float("nan")), s.get("ep_pos", float("nan")),
        "%.3f" % s["rec300_pdfium"] if "rec300_pdfium" in s else "-", "%.3f/%.3f" % (s["ep_pos_pdfium"], s["ep_light_pdfium"]) if "ep_pos_pdfium" in s else "-",
        s["ssim"], s["dE"], 0, s["loss_solid"], s["loss_faint"], s["speckle"],
        ("%.2f" % tm[c]) if c in tm else "-",
        ("PASS (MAD %.1f, p99.9 %.0f; pdf.js %.1f)" % (k["mad"], k["p999"], k["pj"]) if k and k["bad"] == 0 and k["mad"] < 5 and k["pj"] < 5 else ("see note (MAD %.1f, pdf.js %.1f)" % (k["mad"], k["pj"]) if k else "-"))))
hdr = "| cfg | KB med | KB mean | KB p90 | 124-pg doc MB | OCR300 recall (poppler) | Epson pos (poppler) | OCR300 (pdfium) | Epson pos/light (pdfium) | SSIM | dE00 | - | ink-loss solid | ink-loss faint | speckle | enc s | compat |\n|" + "---|" * 17
out = ["## All candidates, 66 corpus pages (colour+gray), cleaner=%s" % TAG, hdr]
order = [c for c in sorted(S, key=lambda c: S[c]["mean"]) if S[c]["n"] >= 60]
for c in order: out.append(row(c))
out.append("\n## Gray-variant candidates, 3 no-colour pages only (pair07_front, pair07_back, image02)\n| cfg | KB mean | OCR300 | SSIM | speckle |\n|---|---|---|---|---|")
for c in sorted(S, key=lambda c: S[c]["mean"]):
    if S[c]["n"] < 60 and c != "E0_literal": out.append("| %s | %.0f | %.3f | %.4f | %.4f |" % (c, S[c]["mean"] / 1e3, S[c].get("rec300", float("nan")), S[c]["ssim"], S[c]["speckle"]))
# same pages, colour candidates
gp = {r["name"] for r in json.load(open(RES / "pages.json")) if r["cls"] == "gray"}
out.append("\n### same 3 gray pages with the colour-capable encoders\n| cfg | KB mean over 3 gray pages |\n|---|---|")
for c in ["E1_j150_q40_420", "E1_j200_q55_420", "E2_p150_k8_km", "E3_c75_q40", "E4_s2", "E4_s7"]:
    v = [r["bytes"] for r in rows if r["cfg"] == c and r["page"] in gp]; out.append("| %s | %.0f |" % (c, sum(v) / len(v) / 1e3))
# class table for finalists
cls = {r["name"]: r["cls"] for r in json.load(open(RES / "pages.json"))}
out.append("\n## Bytes by page class (mean KB)\n| cfg | color (n=%d) | gray (n=%d) |\n|---|---|---|" % (sum(v == "color" for v in cls.values()), sum(v == "gray" for v in cls.values())))
for c in order[:0] + ["E0", "E0_noalpha", "E1_j150_q40_420", "E1_j200_q55_420", "E2_p150_k8_km", "E3_c75_q40", "E3_c50_q40", "E3_base", "E4_s2", "E4_s7"]:
    cs = collections.defaultdict(list)
    for r in rows:
        if r["cfg"] == c and r["page"] != "epson300": cs[cls[r["page"]]].append(r["bytes"])
    out.append("| %s | %.0f | %.0f |" % (c, np.mean(cs["color"]) / 1e3, np.mean(cs["gray"]) / 1e3 if cs["gray"] else float("nan")))
out.append("\n## E0 variants (15 of 66 pages quoted: native/any; all pages for bytes)\n")
for c in ("E0", "E0_noalpha", "E0_literal"):
    b = [r["bytes"] for r in rows if r["cfg"] == c and r["page"] != "epson300"]
    out.append("- %s: n=%d median %.0f KB mean %.0f KB p90 %.0f KB; 124-page doc %.1f MB; encode time (median in shared-load run) %.1f s" % (c, len(b), A.q(b, 50) / 1e3, np.mean(b) / 1e3, A.q(b, 90) / 1e3, np.mean(b) * 124 / 1e6, np.median([r["t_enc"] for r in rows if r["cfg"] == c])))
open(RES / "tables.md", "w").write("\n".join(out)); print("\n".join(out))
# Pareto
for key, lab, fn in (("ssim", "SSIM vs cleaned 300 ppi reference (poppler render)", "ssim"), ("rec300", "OCR word recall, native-300 corpus pages", "rec300"), ("ep_pos", "Epson keymap positional recall (poppler render)", "epson")):
    fig, ax = plt.subplots(figsize=(9, 6)); col = dict(E1="tab:blue", E2="tab:orange", E3="tab:green", E4="tab:red", E0="k")
    for c in order + ["E0"]:
        if c not in S or key not in S[c]: continue
        f = c.split("_")[0]; ax.scatter(S[c]["mean"] / 1e3, S[c][key], c=col.get(f, "gray"), s=22)
        if c in ("E3_c75_q40", "E3_c75_q40_m200", "E1_j150_q40_420", "E1_j200_q55_420", "E4_s2", "E2_p150_k8_km", "E0", "E3_base", "E4_s7"): ax.annotate(c, (S[c]["mean"] / 1e3, S[c][key]), fontsize=7)
    for f, cc in col.items(): ax.scatter([], [], c=cc, label=f)
    ax.set_xlabel("mean KB / page"); ax.set_ylabel(lab); ax.set_xscale("log"); ax.legend(); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(HERE / f"pareto_{fn}.png", dpi=110); plt.close(fig)
