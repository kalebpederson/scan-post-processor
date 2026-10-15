"""Timing + max-abs-diff of scanpp.clean.clean_page vs the reference implementation (clean_page_reference =
research/r2_cleaning/clean.py maths) on corpus pages: .venv/bin/python bench/cmp_clean.py [page names...]"""
import sys, time, json
import cv2, numpy as np
cv2.setNumThreads(1)
from scanpp.clean import clean_page, clean_page_reference

m = json.load(open("research/corpus/manifest.json"))
for nme in sys.argv[1:] or ["pair02_front", "single02", "image01", "single08"]:
    im = cv2.cvtColor(cv2.imread(f"research/corpus/{nme}.jpg"), cv2.COLOR_BGR2RGB)
    dpi = [x for x in m if x["file"] == nme + ".jpg"][0]["dpi"][0]
    for paper in ("flatten", "keep"):
        ro, rn = [], []
        for _ in range(7):            # interleaved, best-of (robust to machine load)
            t = time.process_time(); a = clean_page_reference(im, dpi, paper=paper); ro.append(time.process_time() - t)
            t = time.process_time(); b = clean_page(im, dpi, paper=paper); rn.append(time.process_time() - t)
        d = np.abs(a.astype(int) - b)
        print(f"{nme} {dpi} {paper}: reference {min(ro)*1000:.0f} ms, optimised {min(rn)*1000:.0f} ms ({min(ro)/min(rn):.2f}x); "
              f"max diff {d.max()}, pixels differing {(d > 0).sum()}")
