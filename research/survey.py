#!/usr/bin/env python3
"""Survey samples/: pdfinfo + pdfimages -list + pdftotext per PDF, plus raster images."""
import json, re, subprocess, sys, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
OUT = ROOT / "research" / "survey.json"

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    return r.stdout

def parse_size(s):
    m = re.match(r"([\d.]+)\s*([KMG]?)", s)
    v = float(m.group(1)); u = m.group(2)
    return int(v * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[u])

def survey_pdf(p):
    p = str(p)
    info = run(["pdfinfo", p])
    d = {}
    for line in info.splitlines():
        if ":" in line:
            k, v = line.split(":", 1); d[k.strip()] = v.strip()
    pages = int(d.get("Pages", 0) or 0)
    rec = {"path": str(Path(p).relative_to(ROOT)), "creator": d.get("Creator"), "producer": d.get("Producer"),
           "pages": pages, "page_size": d.get("Page size"), "file_bytes": Path(p).stat().st_size, "images": []}
    if pages == 0:
        rec["class"] = "error"; rec["error"] = "pdfinfo failed"; return rec
    for line in run(["pdfimages", "-list", p]).splitlines()[2:]:
        t = line.split()
        if len(t) < 16: continue
        rec["images"].append({"page": int(t[0]), "type": t[2], "w": int(t[3]), "h": int(t[4]), "color": t[5],
                              "bpc": int(t[7]), "enc": t[8], "ppi": [int(t[12]), int(t[13])], "bytes": parse_size(t[14])})
    txt = run(["pdftotext", "-q", p, "-"])
    rec["text_chars"] = len(re.sub(r"\s", "", txt))
    per = collections.Counter(i["page"] for i in rec["images"] if i["type"] == "image")
    rec["images_per_page"] = [per.get(pg, 0) for pg in range(1, pages + 1)]
    cp = f"{rec['creator']} {rec['producer']}".lower()
    simple = "simple scan" in cp or "simple-scan" in cp
    one_each = all(c == 1 for c in rec["images_per_page"])
    # full-page check: image physical size (px/ppi) covers >=85% of page in both dims
    m = re.match(r"([\d.]+) x ([\d.]+)", rec["page_size"] or "")
    full = False
    if m and rec["images"]:
        pw, ph = float(m.group(1)) / 72, float(m.group(2)) / 72
        full = all(i["w"] / max(i["ppi"][0], 1) >= .85 * pw and i["h"] / max(i["ppi"][1], 1) >= .85 * ph
                   for i in rec["images"] if i["type"] == "image")
    rec["full_page_images"] = full
    rec["simple_scan_tag"] = simple
    rec["class"] = "scan" if (simple or (one_each and full and rec["text_chars"] < 20 * pages)) else "digital"
    return rec

def survey_img(p):
    try:
        im = Image.open(p)
        w, h = im.size
        dpi = im.info.get("dpi")
        dpi = [float(dpi[0]), float(dpi[1])] if dpi else None
        ar = max(w, h) / min(w, h)
        paper = any(abs(ar - a) < 0.04 for a in (11 / 8.5, 1.4142))
        implied = None
        if paper:
            implied = round(min(w, h) / 8.5) if abs(ar - 11/8.5) < 0.04 else round(min(w, h) / 8.27)
        if paper and (dpi and dpi[0] >= 100 or implied and 100 <= implied <= 700):
            kind = "document_scan"
        elif paper:
            kind = "paper_aspect_unknown_dpi"
        else:
            kind = "photo_or_other"
        return {"path": str(Path(p).relative_to(ROOT)), "w": w, "h": h, "mode": im.mode, "format": im.format,
                "dpi": dpi, "aspect": round(ar, 3), "implied_dpi_letter": implied, "bytes": Path(p).stat().st_size, "kind": kind}
    except Exception as e:
        return {"path": str(Path(p).relative_to(ROOT)), "error": str(e), "kind": "error"}

def dist(c): return dict(sorted(c.items(), key=lambda kv: -kv[1]))

def main():
    pdfs = sorted(p for p in SAMPLES.rglob("*") if p.suffix.lower() == ".pdf")
    imgs = sorted(p for p in SAMPLES.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    with ThreadPoolExecutor(16) as ex:
        P = list(ex.map(survey_pdf, pdfs)); I = list(ex.map(survey_img, imgs))
    OUT.write_text(json.dumps({"pdfs": P, "images": I}, indent=1))
    cls = collections.Counter(r["class"] for r in P)
    print("PDF classes:", dict(cls))
    scans = [r for r in P if r["class"] == "scan"]
    ppi = collections.Counter(); enc = collections.Counter(); col = collections.Counter(); psz = collections.Counter()
    tb = 0; tp = 0
    for r in scans:
        psz[r["page_size"]] += r["pages"]; tb += r["file_bytes"]; tp += r["pages"]
        for i in r["images"]:
            if i["type"] != "image": continue
            ppi[i["ppi"][0]] += 1; enc[i["enc"]] += 1; col[f'{i["color"]}/{i["bpc"]}'] += 1
    print("scan ppi:", dist(ppi)); print("scan encodings:", dist(enc)); print("scan color/bpc:", dist(col))
    print("scan page sizes (pages):", dict(list(dist(psz).items())[:8]))
    print(f"scan total bytes {tb} ({tb/1e6:.1f} MB), pages {tp}, bytes/page {tb/max(tp,1):.0f}")
    print("creators(scan):", dist(collections.Counter(f'{r["creator"]}|{r["producer"]}' for r in scans)))
    print("images by kind:", dist(collections.Counter(r["kind"] for r in I)))

main()
