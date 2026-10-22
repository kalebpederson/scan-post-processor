"""usage: why.py IMAGE [DPI] -- for each side, the source line at which boundary.detect_edge_line gives up (or accepts)."""
import sys

import cv2

from scanpp import boundary, geometry

rgb = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_BGR2RGB)
dpi = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
g, d = geometry.prep_gray(rgb, dpi)
mask, info, _ = geometry._edge_mask_work_g(g, d)
small = geometry._work_rgb(rgb, g.shape)
skip = "".join(k[-1] for k in info if k.startswith("coil_"))
print("coil sides:", skip or "-")
src = open(boundary.__file__).read().split("\n")
for side in "TBLR":
    last = {}

    def tr(frame, ev, arg):
        if frame.f_code.co_name == "detect_edge_line":
            if ev == "line":
                last["l"] = frame.f_lineno
            return tr
        return None
    sys.settrace(tr)
    r = boundary.detect_edge_line(small, d, side)
    sys.settrace(None)
    print(side, "ACCEPT" if r else "none", "" if r else f"line {last.get('l')}: {src[last['l'] - 1].strip()[:90]}")
