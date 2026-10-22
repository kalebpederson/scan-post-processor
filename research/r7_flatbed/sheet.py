"""usage: sheet.py OUT.png OLD.png NEW.png ... -- stack, for each (old, new) pair, the 4 corners of both renders
(side by side, old above new) for visual comparison.  Crops are 330 x 330 px at 300 dpi."""
import sys

import cv2
import numpy as np

out = sys.argv[1]
pairs = list(zip(sys.argv[2::2], sys.argv[3::2]))
S = 330
rows = []
for a, b in pairs:
    A, B = cv2.imread(a), cv2.imread(b)
    H, W = A.shape[:2]
    def corners(im):
        cs = [im[:S, :S], im[:S, W - S:], im[H - S:, :S], im[H - S:, W - S:]]
        return np.hstack([cv2.copyMakeBorder(c, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255)) for c in cs])
    rows.append(np.vstack([corners(A), np.full((4, 4 * (S + 4), 3), (255, 0, 0), np.uint8), corners(B),
                           np.full((8, 4 * (S + 4), 3), (0, 128, 0), np.uint8)]))
cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
