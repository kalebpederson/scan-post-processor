"""Halftone benchmark: baseline/variant cleaner on (a) Epson lightest-class OCR recall, (b) stroke-continuity on the Epson light
lines + corpus halftone regions, (c) the same on the r3 1-bit mask (threshold clip(0.35*local_max_7x7(d),20,88)).
Usage: run.py [--params JSON] [--out FILE.json] [--no-ocr] [--save-dir DIR]   (params go to clean_page as kwargs)"""
import sys, os, json, argparse, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2, numpy as np
cv2.setNumThreads(4)
import metrics as M, regions as R
from scanpp.clean import clean_page

def r3_mask(rgb):
    d = 255.0 - rgb.min(-1).astype(np.float32)
    t = np.clip(0.35 * cv2.dilate(d, np.ones((7, 7), np.uint8)), 20, 88)
    return d >= t

def mask_as_rgb(m): return np.repeat(np.where(m, 0, 255).astype(np.uint8)[..., None], 3, 2)

def region_scores(rgb, dpi, boxes, out, ids, sigma_in=0.008, pad_in=0.04):
    res = {}
    cache = {}
    for i, box in zip(ids, boxes):
        x0, y0, x1, y1 = M.pad_box(box, dpi, rgb.shape, pad_in)
        crop = rgb[y0:y1, x0:x1]
        S, _ = M.ref_strokes(crop, dpi, sigma_in=sigma_in)
        oc = out[y0:y1, x0:x1]
        res[i] = dict(out=M.score(S, oc, dpi), mask=M.score(S, mask_as_rgb(r3_mask(oc)), dpi))
    return res

def agg(res, key):
    rows = [v[key] for v in res.values() if v.get(key)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--params", default="{}"); ap.add_argument("--out"); ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--save-dir"); a = ap.parse_args()
    P = json.loads(a.params); out = {}
    # Epson
    import epson_ocr as O
    rgb = O.epson300(); t = time.time(); cl = clean_page(rgb, 300, **P); out["epson_clean_s"] = time.time() - t
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True); cv2.imwrite(os.path.join(a.save_dir, "epson_clean.png"), cl[..., ::-1])
    if not a.no_ocr:
        for name, im in (("raw", rgb), ("clean", cl)):
            r = O.score_rgb(im); r.pop("hits"); out["ocr_" + name] = r
    L = json.load(open(os.path.join(HERE, "epson_lines.json")))["lines"]
    er = region_scores(rgb, 300, [l["box"] for l in L], cl, [f"line{i}" for i in range(len(L))], sigma_in=0.004, pad_in=0.01)
    out["epson_lines"] = dict(out=agg(er, "out"), mask=agg(er, "mask"), per=er)
    # corpus regions
    pages = {}; cr = {}
    for r in R.regions():
        key = r.get("file") or (r["pdf"], r["img"])
        if key not in pages:
            pg = R.load_page(r); pages[key] = (pg, clean_page(pg, r["dpi"], **P))
        pg, c = pages[key]
        cr.update(region_scores(pg, r["dpi"], [r["box"]], c, [r["id"]]))
        if a.save_dir: cv2.imwrite(os.path.join(a.save_dir, r["id"] + ".png"), c[r["box"][1]:r["box"][1] + r["box"][3], r["box"][0]:r["box"][0] + r["box"][2]][..., ::-1])
    out["corpus_regions"] = dict(out=agg(cr, "out"), mask=agg(cr, "mask"), per=cr)
    if a.out: json.dump(out, open(a.out, "w"), indent=1)
    f = lambda d: {k: round(v, 3) for k, v in d.items()}
    for k in ("ocr_raw", "ocr_clean"):
        if k in out: print(k, {kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in out[k].items()})
    for k in ("epson_lines", "corpus_regions"):
        print(k, "out ", f(out[k]["out"])); print(k, "mask", f(out[k]["mask"]))
if __name__ == "__main__":
    main()
