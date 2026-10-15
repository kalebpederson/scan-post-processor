"""Round 2 table: study metrics + m2 metrics per config (mean over pages)."""
import glob, collections
from config import *
import aggregate as A
RES = WORK / "res"
rows = A.load(); S = A.summarize(rows)
M = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(str(RES / "m2" / "*" / "*.json")):
    c = f.split("/")[-2]; r = json.load(open(f))
    for k, v in r.items():
        if v is not None: M[c][k].append(v)
pd = collections.defaultdict(dict)
for f in glob.glob(str(RES / "pdfium_ocr" / "*" / "*.json")): pd[f.split("/")[-2]][f.split("/")[-1][:-5]] = json.load(open(f))
def mm(c, k): v = M[c].get(k); return float(np.mean(v)) if v else float("nan")
def table(cfgs):
    out = ["| cfg | KB mean | OCR300 | Epson pos/light | SSIM | dE | colour-ID changed % (comp>=100px) | hue-shift % | area-wtd % | swap % | hl ink dE | hl near-ink dE | pencil wt (mean/min) | speckle |", "|" + "---|" * 14]
    for c in cfgs:
        if c not in S: continue
        s = S[c]
        out.append("| %s | %.0f | %.3f | %.3f/%.3f | %.4f | %.1f | %.1f | %.1f | %.1f | %.2f | %.1f | %.1f | %.2f/%.2f | %.4f |" % (
            c, s["mean"] / 1e3, s.get("rec300", float("nan")), s.get("ep_pos", float("nan")), s.get("ep_light", float("nan")), s["ssim"], s["dE"],
            100 * mm(c, "cid100_changed"), 100 * mm(c, "cid100_hue"), 100 * mm(c, "cid_area_changed"), 100 * mm(c, "cid_swap"),
            mm(c, "hl_ink"), mm(c, "hl_near"), mm(c, "pencil_ratio"), mm(c, "pencil_min"), s["speckle"]))
    return "\n".join(out)
if __name__ == "__main__":
    import sys
    cfgs = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(c for c in S if c in M and S[c]["n"] >= 60)
    t = table(cfgs); print(t); open(RES / "tables_r2.md", "w").write(t)
