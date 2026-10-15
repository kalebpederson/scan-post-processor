"""Before/after panels -> bench/panels/halftone_*.png.  'before' = clean_page(desc_sigma_in=0) (the previous cleaner), 'after' = defaults.
Columns: raw | before | after | r3 mask before | r3 mask after (mask = d >= clip(0.35*local_max_7x7(d), 20, 88) on the cleaned image)."""
import sys, os, json
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "tests"))
import cv2, numpy as np
cv2.setNumThreads(4)
import regions as R, epson_ocr as O
from run import r3_mask, mask_as_rgb
from scanpp.clean import clean_page
OUT = os.path.join(ROOT, "bench", "panels")
BEFORE = dict(desc_sigma_in=0)

def tile(img, scale, label=None):
    t = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale != 1 else img.copy()
    t = cv2.copyMakeBorder(t, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(200, 200, 200))
    if label: cv2.putText(t, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 0, 0), 1, cv2.LINE_AA)
    return t

def row(rgb, box, dpi, cl_b, cl_a, scale, name, crop_w=None):
    x, y, w, h = box
    if crop_w: w = min(w, crop_w)
    sl = (slice(y, y + h), slice(x, x + w))
    ims = [rgb[sl], cl_b[sl], cl_a[sl], mask_as_rgb(r3_mask(cl_b[sl])), mask_as_rgb(r3_mask(cl_a[sl]))]
    labs = [name, "before", "after", "mask before", "mask after"]
    return np.hstack([tile(i, scale, l) for i, l in zip(ims, labs)])

def stack(rows, path):
    W = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    cv2.imwrite(path, np.vstack(rows)[..., ::-1]); print(path, np.vstack(rows).shape)

def main():
    rgb = O.epson300(); b = clean_page(rgb, 300, **BEFORE); a = clean_page(rgb, 300)
    L = json.load(open(os.path.join(HERE, "epson_lines.json")))["lines"]
    manual = [l for l in L if l.get("manual")]
    show = [dict(box=[738, 2276, 300, 38], text="Show bookmarks?")]
    pick = manual + sorted([l for l in L if not l.get("manual")], key=lambda l: l["dark"])[:6]
    stack([row(rgb, l["box"], 300, b, a, 1, l["text"][:18]) for l in pick], os.path.join(OUT, "halftone_epson_lightest_1x.png"))
    stack([row(rgb, l["box"], 300, b, a, 2, l["text"][:14], crop_w=150) for l in pick[:5]], os.path.join(OUT, "halftone_epson_lightest_2x.png"))
    rows1, rows2 = [], []
    for r in R.regions():
        pg, dpi, box = R.load_region(r)
        pb, pa = clean_page(pg, dpi, **BEFORE), clean_page(pg, dpi)
        rows1.append(row(pg, box, dpi, pb, pa, 1, r["id"]))
        rows2.append(row(pg, [box[0], box[1], min(box[2], 140), min(box[3], 100)], dpi, pb, pa, 2, r["id"]))
    stack(rows1[:4], os.path.join(OUT, "halftone_corpus_regions_a_1x.png")); stack(rows1[4:], os.path.join(OUT, "halftone_corpus_regions_b_1x.png"))
    stack(rows2[:4], os.path.join(OUT, "halftone_corpus_regions_a_2x.png")); stack(rows2[4:], os.path.join(OUT, "halftone_corpus_regions_b_2x.png"))
    # no-regression regions: bleed-through (negative) + pencil + light print labelled by R2
    LAB = json.load(open(os.path.join(ROOT, "research/r2_cleaning/labels.json")))
    sel = [x for x in LAB["neg"] if x["id"] in ("bleed_01", "bleed_05", "bleed_pink")] + [x for x in LAB["pos"] if x["id"] in ("pencil_pair03_a", "pencil_image04", "pencil_epson_hatch", "grayline_pair06")]
    rows = []
    for r in sel:
        f = os.path.join(ROOT, r["file"]); pg = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        pb, pa = clean_page(pg, r["dpi"], **BEFORE), clean_page(pg, r["dpi"])
        x, y, w, h = r["box"]; pad = 20
        rows.append(row(pg, [max(0, x - pad), max(0, y - pad), w + 2 * pad, h + 2 * pad], r["dpi"], pb, pa, 1 if w > 200 else 2, r["id"]))
    stack(rows, os.path.join(OUT, "halftone_noregression_bleed_pencil.png"))
    # synthetic
    from halftone_synth import halftone, render_text
    rs = []
    for lpi, tone, sc, th, bl in ((150, 0.30, 1.2, 2, 1.0), (175, 0.25, 1.0, 2, 0.9)):
        cov = render_text(scale=sc, thickness=th); g = halftone(cov * tone, lpi=lpi, blur_px=bl, dot_value=60); pg = np.repeat(g[..., None], 3, 2)
        rs.append(row(pg, [40, 40, 330, 120], 300, clean_page(pg, 300, **BEFORE), clean_page(pg, 300), 1, f"synth {lpi}lpi"))
    stack(rs, os.path.join(OUT, "halftone_synthetic.png"))
main()
