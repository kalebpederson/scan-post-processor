"""Does the corpus / samples contain standard 3-hole-punch pages (3 large holes ~4.25 in apart, backing visible)?  Re-runs the row
detector with relaxed constants (>= 3 holes, 0.2-0.45 in across, pitch 3.5-5 in, centre up to 0.6 in from the border).
usage: probe_3hole.py SRC_DIR NAME   -> prints the number of pages that fire (and file index / page)."""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r7_flatbed"))
from eval_edges import pages_of, EXT_IMG
from scanpp import geometry


def work(a):
    fid, p = a
    geometry.HOLE_MIN_COUNT = 3
    geometry.HOLE_DIAM_IN = (0.2, 0.45)
    geometry.HOLE_PITCH_IN = (3.5, 5.0)
    geometry.HOLE_MAX_DEPTH_IN = 0.6
    geometry.HOLE_OCCUPANCY = 0.6
    out = []
    for i, rgb, dpi, err in pages_of(Path(p)):
        if rgb is None:
            continue
        g, d = geometry.prep_gray(rgb, dpi)
        _, info, _ = geometry._edge_mask_work_g(g, d)
        if any(k.startswith("holes_") for k in info):
            out.append((fid, i, {k: v for k, v in info.items() if k.startswith("holes_")}))
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1])
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXT_IMG | {".pdf"} and not p.name.startswith("contact_"))
    with ProcessPoolExecutor(8) as ex:
        res = [r for rs in ex.map(work, list(enumerate(files)), chunksize=1) for r in rs]
    print(sys.argv[2], len(res), res[:20])
