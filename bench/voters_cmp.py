"""Compare individual skew voters (package vs research/r4 geometry.py) on corpus pages: .venv/bin/python bench/voters_cmp.py [names...]"""
import sys, importlib.util, json, cv2, numpy as np
cv2.setNumThreads(1)
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("old_geo", "research/r4_geometry/geometry.py"); og = importlib.util.module_from_spec(spec); spec.loader.exec_module(og)
from scanpp import geometry as ng
m = json.load(open("research/corpus/manifest.json"))
names = sys.argv[1:] or ["single05", "pair01_front", "image02"]
for n in names:
    im = cv2.cvtColor(cv2.imread(f"research/corpus/{n}.jpg"), cv2.COLOR_BGR2RGB)
    dpi = [x for x in m if x["file"] == n + ".jpg"][0]["dpi"][0] if n != "single05" else 150
    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    e0, e1 = og.skew_estimates(gray, dpi), ng.skew_estimates(gray, dpi)
    print(n, dpi, "old", {k: round(v, 3) for k, v in e0.items()}, og.estimate_skew(gray, dpi))
    print(" " * len(n), "   new", {k: round(v, 3) for k, v in e1.items()}, ng.estimate_skew(gray, dpi))
