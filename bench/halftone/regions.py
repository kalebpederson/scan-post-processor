"""Loader for bench/halftone/regions.json: returns (rgb_page, dpi, box) for a region, extracting scan pages from samples/ with pdfimages
(read-only) into a cache dir (default $SCANPP_HT_CACHE or /tmp/scanpp_halftone_cache)."""
import os, json, subprocess, glob, hashlib
import cv2, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.environ.get("SCANPP_HT_CACHE", "/tmp/scanpp_halftone_cache")

def regions():
    return json.load(open(os.path.join(HERE, "regions.json")))["regions"]

def load_page(r):
    if "file" in r:
        return cv2.cvtColor(cv2.imread(os.path.join(ROOT, r["file"])), cv2.COLOR_BGR2RGB)
    key = hashlib.md5(r["pdf"].encode()).hexdigest()[:8]
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"{key}-{r['img']:03d}.png")
    if not os.path.exists(f):
        subprocess.run(["pdfimages", "-f", "1", "-l", "3", "-png", os.path.join(ROOT, r["pdf"]), os.path.join(CACHE, key)], check=True)
    return cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)

def load_region(r, pad=0):
    pg = load_page(r); x, y, w, h = r["box"]
    return pg, float(r["dpi"]), (x, y, w, h)
