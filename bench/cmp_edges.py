"""scanpp.geometry.clean_edges / paper_colour vs the research implementation (must be bit-identical) on the whole corpus,
plus timing: .venv/bin/python bench/cmp_edges.py"""
import sys, time, json, importlib.util
import cv2, numpy as np
cv2.setNumThreads(1)
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("old_geo", "research/r4_geometry/geometry.py")
old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
from scanpp import geometry as new

def best(f, n=7):
    r = []
    for _ in range(n):
        t = time.process_time(); v = f(); r.append(time.process_time() - t)
    return min(r) * 1000, v

m = json.load(open("research/corpus/manifest.json"))
bad = 0
for x in m:
    nme, dpi = x["file"], x["dpi"][0]
    im = cv2.cvtColor(cv2.imread("research/corpus/" + nme), cv2.COLOR_BGR2RGB)
    for fill in (None, (255, 255, 255)):
        a, ia = old.clean_edges(im, dpi, fill); b, ib = new.clean_edges(im, dpi, fill)
        same = np.array_equal(a, b) and abs(ia["painted_frac"] - ib["painted_frac"]) < 1e-12 and all(ia[k] == ib[k] for k in ia if k != "painted_frac")
        bad += not same
    assert old.paper_colour(im) == new.paper_colour(im)
    if nme in ("pair02_front.jpg", "single02.jpg"):
        print(nme, "research %.0f ms, optimised %.0f ms" % (best(lambda: old.clean_edges(im, dpi))[0], best(lambda: new.clean_edges(im, dpi))[0]))
print("pages", len(m), "mismatches", bad)
