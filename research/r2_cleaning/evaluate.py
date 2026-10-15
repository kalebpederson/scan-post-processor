"""Metric harness.  evaluate(configs) -> per-config aggregate dict.  configs: list of (name, method, kwargs[, paper])."""
import sys, json, time, os; sys.path.insert(0, '.')
from common import *
from bleedlib import accepted_pairs, bleed_masks
from clean import clean_page
from multiprocessing import Pool

LAB = json.load(open(HERE / "labels.json"))
COLORED_PAPER = {"single08", "image01", "single38"}

def run_method(rgb, dpi, method, kw, paper="flatten"):
    if method == "b0": return b0(rgb, **kw)
    return clean_page(rgb, dpi, paper=paper, **kw)

def lab_pix(rgb_px):  # Nx3 uint8 -> L,C,h
    L, C, h = lab_lch(rgb_px.reshape(-1, 1, 3)); return L.ravel(), C.ravel(), h.ravel()

def dE_white(rgb_px):
    lab = cv2.cvtColor(rgb_px.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).astype(np.float32).reshape(-1, 3)
    lab[:, 0] *= 100 / 255; lab[:, 1:] -= 128
    return np.sqrt((100 - lab[:, 0]) ** 2 + lab[:, 1] ** 2 + lab[:, 2] ** 2)

def task(args):
    path, dpi, name, kinds, bleed_pairs, configs = args
    rgb = load_rgb(path); h, w = rgb.shape[:2]
    bg = ref_background(rgb, dpi); d, n = darkness(rgb, dpi, bg)
    ds = cv2.GaussianBlur(d, (0, 0), max(0.6, 0.006 * dpi))
    k = lambda r_in: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(round(r_in * dpi))) + 1,) * 2)
    valid = np.zeros((h, w), bool); valid[int(.03 * h):int(.97 * h), int(.12 * w):int(.96 * w)] = True
    nat = {}
    if "page" in kinds:
        P = (ds < 0.035) & ~(cv2.dilate((ds > 0.08).astype(np.uint8), k(0.04)) > 0) & valid
        Bk = (ds < 0.15) & ~(cv2.dilate((ds > 0.15).astype(np.uint8), k(0.04)) > 0) & valid
        L, C, hh = lab_lch(rgb)
        Ptint = P & (C > 8) if name in COLORED_PAPER else None
        dark = valid & (rgb.max(-1) <= 90) & (C < 30)
        colr = valid & (C > 25) & (d > 0.2)
        ci = np.flatnonzero(colr.ravel()); rng = np.random.default_rng(1)
        if len(ci) > 30000: ci = rng.choice(ci, 30000, replace=False)
        cy, cx = np.unravel_index(ci, (h, w))
        if len(ci) == 0: ci = np.array([0]); cy, cx = np.array([0]), np.array([0]); hasc = False
        else: hasc = True
        nat_c = rgb[cy, cx]; Lc, Cc, hc = lab_pix(nat_c)
        ptarget = np.median(rgb[P].reshape(-1, 3), 0) if P.sum() > 100 else np.array([255, 255, 255.])
    regs = [r for r in LAB["pos"] + LAB["neg"] if str(ROOT / r["file"]) == str(path) or os.path.abspath(ROOT / r["file"]) == os.path.abspath(path)]
    regm = []
    for r in regs:
        x, y, bw_, bh_ = r["box"]; m = np.zeros((h, w), bool); m[y:y + bh_, x:x + bw_] = True
        # stroke CORE: darkest pixels of the region: ds >= max(0.04, 0.5*p95(ds | ds>=0.03)) -> excludes antialias skirts / paper
        v = ds[m & (ds >= 0.03)]
        thr = max(0.04, 0.5 * float(np.percentile(v, 95))) if len(v) > 20 else 1.0
        S = m & (ds >= thr) & (ds <= r["dmax"])
        regm.append((r, S))
    bl = []
    for p in bleed_pairs:
        bl.append((p, bleed_masks(p, rgb, d)))
    nat_dE = lambda S: None
    out = {}
    for cfg in configs:
        cname, method, kw = cfg[0], cfg[1], cfg[2]; paper = cfg[3] if len(cfg) > 3 else "flatten"
        t0 = time.perf_counter(); o = run_method(rgb, dpi, method, kw, paper); dt = time.perf_counter() - t0
        res = dict(time=dt, mp=h * w / 1e6)
        ol = o.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        if "page" in kinds:
            white = o.min(-1) >= 250
            res["white"] = (float(white[P].mean()) if P.sum() > 100 else None, int(P.sum()))
            res["bk"] = (float(white[Bk].mean()) if Bk.sum() > 100 else None, int(Bk.sum()))
            if Ptint is not None and Ptint.sum() > 500: res["white_tint"] = float(white[Ptint].mean())
            if P.sum() > 100:
                lab_o = cv2.cvtColor(o[P][:: max(1, P.sum() // 50000)].reshape(-1, 1, 3), cv2.COLOR_RGB2Lab).astype(np.float32).reshape(-1, 3)
                tgt = cv2.cvtColor(np.uint8(ptarget).reshape(1, 1, 3), cv2.COLOR_RGB2Lab).astype(np.float32).reshape(3)
                res["keep_dE"] = (float(np.mean(np.sqrt(((lab_o - tgt) * np.array([100 / 255, 1, 1])) ** 2).sum(1))), float(lab_o[:, 0].std() * 100 / 255))
            res["inkarea"] = (int(((ol < 128) & valid).sum()), int(((rgb.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32) < 128) & valid).sum()))
            res["ink_hist"] = np.bincount(np.clip(ol[dark].astype(int), 0, 255), minlength=256).tolist()
            oc = o[cy, cx]; Lo, Co, ho = lab_pix(oc)
            dh = np.abs((ho - hc + 180) % 360 - 180)
            if hasc: res["color"] = dict(dh=dh.tolist()[:5000], cr=(Co / np.maximum(Cc, 1)).tolist()[:5000], Lo=Lo.tolist()[:5000], Ln=Lc.tolist()[:5000], Co=Co.tolist()[:5000], Cn=Cc.tolist()[:5000])
        rr = []
        for r, S in regm:
            if S.sum() < 20: rr.append((r["id"], r["cls"], 0, 0, 0, 0)); continue
            px = o[S]; dE = dE_white(px)
            ndE = dE_white(rgb[S])
            rr.append((r["id"], r["cls"], int(S.sum()), int((dE >= 6).sum()), float(dE.mean()), float(ndE.mean())))
        res["regions"] = rr
        bb = []
        for p, m in bl:
            vis = m["visible"]; reg = m["region"]
            if vis.sum() == 0: continue
            nonwhite = (o.min(-1) < 250)
            bb.append(dict(key=f"{p['src'][-30:]}:{p['front']}<-{p['back']}", nvis=int(vis.sum()), nreg=int(reg.sum()),
                           surv=float(nonwhite[vis].mean()), dark_vis=float((1 - ol[vis] / 255).mean()), dark_reg=float((1 - ol[reg] / 255).mean()),
                           nat_dark_vis=float(d[vis].mean()), surv_dE=float((dE_white(o[vis][:: max(1, vis.sum() // 20000)]) >= 6).mean())))
        res["bleed"] = bb
        out[cname] = res
    return name, out
