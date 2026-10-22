"""usage: page_probe.py FILE PAGE [OUT_PREFIX] -- edge info + paper boundaries of one page; optionally saves the decoded page."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_edges import pages_of  # noqa: E402

from scanpp import geometry  # noqa: E402

f, page = sys.argv[1], int(sys.argv[2])
for i, rgb, dpi, err in pages_of(Path(f)):
    if i != page:
        continue
    g, d = geometry.prep_gray(rgb, dpi)
    mask, info, _ = geometry._edge_mask_work_g(g, d)
    print({k: round(v, 3) for k, v in info.items()}, rgb.shape, dpi)
    for r in geometry._paper_boundaries(rgb, g, d, info):
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "depth"})
    if len(sys.argv) > 3:
        cv2.imwrite(sys.argv[3] + ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
