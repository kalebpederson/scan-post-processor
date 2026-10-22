"""Print rows (distance from the border) of rendered pages where >= frac of the columns are darker than thr.
usage: rowscan.py THR FRAC file.png ..."""
import sys

import cv2

thr, frac = int(sys.argv[1]), float(sys.argv[2])
for f in sys.argv[3:]:
    g = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    for name, band in (("B", g[H - 200:][::-1]), ("T", g[:200]), ("L", g[:, :200].T), ("R", g[:, W - 200:][::-1].T)):
        fr = (band[:, 300:-300] < thr).mean(axis=1)
        rows = [(i, round(float(v), 2)) for i, v in enumerate(fr) if v >= frac]
        print(f.split("/")[-1], name, rows[:6])
