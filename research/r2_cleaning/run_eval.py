import sys, json, pickle, time; sys.path.insert(0, '.')
from common import *
from evaluate import *
from bleedlib import accepted_pairs

def build_tasks(configs, quick=False):
    tasks = []
    items = corpus_items()
    if quick: items = [it for it in items if it[0] in QUICK]
    for name, path, dpi in items:
        tasks.append((path, dpi, name, ("page",), [], configs))
    tasks.append((str(SCANS / "epson_300dpi.png"), 300, "epson", ("page",), [], configs))
    byf = {}
    for p in accepted_pairs(): byf.setdefault(p["front_file"], []).append(p)
    for f, ps in byf.items():
        if quick and len(tasks) > 40 and False: continue
        pth = str(HERE / f) if not f.startswith("/") else f
        tasks.append((pth, ps[0]["dpi"], "bl:" + f.split("pages/")[1], ("bleed",), [dict(p, front_file=pth, back_file=str(HERE / p["back_file"])) for p in ps], configs))
    return tasks
QUICK = {"pair03_front", "pair06_front", "image04", "single09", "single06", "pair07_front", "single28", "single08", "image01", "single38",
         "pair02_front", "single20", "single13", "image03", "single11", "pair09_front"}

def evaluate(configs, quick=False, procs=16):
    tasks = build_tasks(configs, quick)
    with Pool(procs) as p: res = p.map(task, tasks, chunksize=1)
    return aggregate(configs, res)

def aggregate(configs, res):
    out = {}
    for cfg in configs:
        c = cfg[0]; A = {}
        pages = [(n, r[c]) for n, r in res if not n.startswith("bl:")]
        wp = [(n, r) for n, r in pages if n not in COLORED_PAPER and n != "epson" and r.get("white") and r["white"][0] is not None]
        A["white_whitepaper"] = float(np.mean([r["white"][0] for n, r in wp]))
        A["white_whitepaper_min"] = float(np.min([r["white"][0] for n, r in wp]))
        A["bk_clean"] = float(np.mean([r["bk"][0] for n, r in pages if n != "epson" and r.get("bk") and r["bk"][0] is not None]))
        A["white_tint"] = {n: r["white_tint"] for n, r in pages if "white_tint" in r}
        A["white_colored_pages"] = {n: r["white"][0] for n, r in pages if n in COLORED_PAPER and r["white"][0] is not None}
        kd = [r["keep_dE"] for n, r in pages if r.get("keep_dE") and n != "epson"]
        A["keep_dE_mean"] = float(np.mean([k[0] for k in kd])); A["keep_L_std"] = float(np.mean([k[1] for k in kd]))
        ia = np.array([r["inkarea"] for n, r in pages if n != "epson"], float); A["ink_area_ratio"] = float(ia[:, 0].sum() / ia[:, 1].sum())
        H = np.sum([r["ink_hist"] for n, r in pages], 0).astype(float); cs = np.cumsum(H) / H.sum()
        A["ink_med"] = int(np.searchsorted(cs, .5)); A["ink_p90"] = int(np.searchsorted(cs, .9)); A["ink_mean"] = float((H * np.arange(256)).sum() / H.sum())
        pc = [r for n, r in pages if "color" in r and n not in COLORED_PAPER and n != "epson"]   # coloured-paper pages excluded: paper itself counts as "coloured"
        dh = np.concatenate([r["color"]["dh"] for r in pc]); cr = np.concatenate([r["color"]["cr"] for r in pc])
        Lo = np.concatenate([r["color"]["Lo"] for r in pc]); Ln = np.concatenate([r["color"]["Ln"] for r in pc])
        A["hue_shift_mean"] = float(dh.mean()); A["hue_shift_p90"] = float(np.percentile(dh, 90))
        A["chroma_ratio_med"] = float(np.median(cr)); A["chroma_ratio_p10"] = float(np.percentile(cr, 10)); A["color_dL_med"] = float(np.median(Lo - Ln))
        # regions
        reg = {}
        for n, r in res:
            for (rid, cls, cnt, ret, ode, nde) in r[c]["regions"]: reg[rid] = (cls, cnt, ret, ode, nde)
        pos = {k: v for k, v in reg.items() if v[0] != "bleed" and v[1] > 0}
        neg = {k: v for k, v in reg.items() if v[0] == "bleed" and v[1] > 0}
        A["pos_ret"] = float(np.mean([v[2] / v[1] for v in pos.values()]))
        A["pos_contrast_ratio"] = float(np.mean([v[3] / max(v[4], 1e-3) for v in pos.values()]))
        A["pos_out_dE"] = float(np.mean([v[3] for v in pos.values()])); A["pos_nat_dE"] = float(np.mean([v[4] for v in pos.values()]))
        A["pos_by_class"] = {}
        for cl in sorted({v[0] for v in pos.values()}): A["pos_by_class"][cl] = float(np.mean([v[2] / v[1] for v in pos.values() if v[0] == cl]))
        A["pos_each"] = {k: round(v[2] / v[1], 3) for k, v in pos.items()}
        A["neg_surv_regions"] = float(np.mean([v[2] / v[1] for v in neg.values()])) if neg else None
        A["neg_each"] = {k: round(v[2] / v[1], 3) for k, v in neg.items()}
        bl = [b for n, r in res if n.startswith("bl:") for b in r[c]["bleed"]]
        if bl:
            nv = np.array([b["nvis"] for b in bl]); 
            A["bleed_surv_pooled"] = float((np.array([b["surv"] for b in bl]) * nv).sum() / nv.sum())
            A["bleed_surv_macro"] = float(np.mean([b["surv"] for b in bl]))
            A["bleed_surv_dE_macro"] = float(np.mean([b["surv_dE"] for b in bl]))
            A["bleed_dark_vis"] = float(np.mean([b["dark_vis"] for b in bl])); A["bleed_dark_region"] = float(np.mean([b["dark_reg"] for b in bl]))
            A["bleed_nat_dark_vis"] = float(np.mean([b["nat_dark_vis"] for b in bl])); A["n_bleed_instances"] = len(bl)
        tt = [(r[c]["time"], r[c]["mp"]) for n, r in res if not n.startswith("bl:")]
        A["ms_per_mp_parallel_load"] = float(sum(t for t, m in tt) / sum(m for t, m in tt) * 1000)
        out[c] = A
    return out, res

if __name__ == "__main__":
    cfgs = [("B0", "b0", {}), ("M2_default", "clean", {}), ("M2_keep", "clean", {}, "keep")]
    t = time.time(); agg, res = evaluate(cfgs)
    print("elapsed", round(time.time() - t, 1))
    for k, v in agg.items():
        print(k); print(json.dumps({a: b for a, b in v.items() if a not in ("pos_each", "neg_each")}, indent=1)[:3000])
        print(v["pos_each"]); print(v["neg_each"])
