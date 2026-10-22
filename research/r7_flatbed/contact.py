"""usage: contact.py CHANGED_DIR OUT_PREFIX [N_PER_SHEET=12] [--sample K] [--seed S]
Tiles the before|after crops written by eval_edges.py into contact sheets (2 columns), labelled with the file stem."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np

d, prefix = Path(sys.argv[1]), sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 12
files = sorted(d.glob("*.png"))
if "--sample" in sys.argv:
    k = int(sys.argv[sys.argv.index("--sample") + 1])
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    files = sorted(random.Random(seed).sample(files, min(k, len(files))))
TW, TH = 900, 200
tiles = []
for f in files:
    im = cv2.imread(str(f))
    s = min(TW / im.shape[1], TH / im.shape[0])
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = np.full((TH + 16, TW, 3), 255, np.uint8)
    t[16:16 + im.shape[0], :im.shape[1]] = im
    cv2.putText(t, f.stem, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 128, 0)))
for n in range(0, len(tiles), per):
    chunk = tiles[n:n + per]
    while len(chunk) % 2:
        chunk.append(np.full_like(tiles[0], 200))
    rows = [np.hstack(chunk[i:i + 2]) for i in range(0, len(chunk), 2)]
    cv2.imwrite(f"{prefix}_{n // per + 1}.png", np.vstack(rows))
print(len(files), "tiles")
