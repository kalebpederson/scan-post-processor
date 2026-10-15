"""Step 3: aggregate per-(cfg,page) JSONs into tables (results.json + markdown tables)."""
import glob, statistics as st
from config import *
RES = WORK / "res"
INFO = {r["name"]: r for r in json.load(open(RES / "pages.json"))}
FULLDOC = 124

def load():
    rows = []
    for f in glob.glob(str(RES / "cfg" / "*" / "*.json")): rows.append(json.load(open(f)))
    return rows

def q(x, p):
    x = sorted(x)
    if not x: return float("nan")
    k = (len(x) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(x) - 1); return x[lo] + (x[hi] - x[lo]) * (k - lo)
def mean(x): x = [v for v in x if v == v]; return sum(x) / len(x) if x else float("nan")

def summarize(rows, pages_filter=None):
    """returns dict cfg -> summary over pages in filter (default all non-epson corpus pages + separately epson)."""
    by = {}
    for r in rows: by.setdefault(r["cfg"], []).append(r)
    out = {}
    for cfg, rs in by.items():
        main = [r for r in rs if r["page"] != "epson300" and (pages_filter is None or pages_filter(r))]
        ep = [r for r in rs if r["page"] == "epson300"]
        s = dict(cfg=cfg, n=len(main))
        b = [r["bytes"] for r in main]
        s.update(median=q(b, 50), mean=mean(b), p90=q(b, 90), doc124=mean(b) * FULLDOC)
        for k in ("ssim", "dE", "loss_solid", "loss_faint", "speckle"):
            v = [r["fid"][k] for r in main if "fid" in r]; s[k] = mean(v)
        s["t_enc"] = st.median([r["t_enc"] for r in main]) if main else float("nan")
        rc = [r["ocr"]["recall"] for r in main if "ocr" in r and r["ocr"].get("nref", 0) >= 20]
        s["rec300"] = mean(rc); s["n_rec"] = len(rc)
        if ep:
            e = ep[0]; s["epson_bytes"] = e["bytes"]
            if "ocr" in e: s.update(ep_pos=e["ocr"]["pos_recall"], ep_light=e["ocr"]["pos_light"], ep_ms=e["ocr"]["ms_recall"], ep_mid=e["ocr"]["pos_mid"], ep_bold=e["ocr"]["pos_bold"])
        out[cfg] = s
    return out

if __name__ == "__main__":
    rows = load(); S = summarize(rows)
    json.dump(S, open(RES / "summary_all.json", "w"), indent=1)
    print("%-22s %3s %8s %8s %8s | %6s %6s %5s %6s %7s | rec300 ep_pos ep_light | t" % ("cfg", "n", "median", "mean", "p90", "ssim", "dE", "lossS", "lossF", "speck"))
    for c in sorted(S, key=lambda c: S[c]["mean"]):
        s = S[c]
        print("%-22s %3d %8.0f %8.0f %8.0f | %6.4f %6.2f %5.3f %6.3f %7.5f | %.3f %.3f %.3f | %.1f" % (c, s["n"], s["median"], s["mean"], s["p90"], s["ssim"], s["dE"], s["loss_solid"], s["loss_faint"], s["speckle"], s.get("rec300", float("nan")), s.get("ep_pos", float("nan")), s.get("ep_light", float("nan")), s["t_enc"]))
