"""Which pages does the r9 binding-hole rule change?  Edge-clean each page twice in one process: `geometry.BINDING_HOLES` off
(previous behaviour) and on.

usage: eval_holes.py SRC_DIR NAME [--procs 16]
Writes research/r9_holes/eval_NAME.json (aggregate; pages keyed by file index, never by file name) and, local only,
research/r9_holes/work/files_NAME.json (index -> path) plus a tile of every firing page in research/r9_holes/work/changed_NAME/:
the border strip (full height, scaled) before | after | newly painted pixels in red.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "r7_flatbed"))
from eval_edges import pages_of, EXT_IMG  # noqa: E402
from scanpp import geometry  # noqa: E402

HERE = Path(__file__).resolve().parent


def work(args):
    fid, path, outdir = args
    out = []
    for i, rgb, dpi, err in pages_of(Path(path)):
        if rgb is None:
            out.append(dict(fid=fid, page=i, err=err))
            continue
        try:
            geometry.BINDING_HOLES = False
            old, _ = geometry.clean_edges(rgb, dpi)
            geometry.BINDING_HOLES = True
            new, info = geometry.clean_edges(rgb, dpi)
            sides = sorted(k[-1] for k in info if k.startswith("holes_"))
            diff = (old != new).any(axis=2)
            rec = dict(fid=fid, page=i, dpi=round(float(dpi), 1), fired=bool(sides), changed=bool(diff.any()),
                       holes={s: info["holes_" + s] for s in sides}, coil=sorted(k[-1] for k in info if k.startswith("coil_")))
            if sides:
                rec["changed_frac"] = float(diff.mean())
                H, W = diff.shape
                w = int(0.9 * dpi)
                s = sides[0]
                sl = (slice(None), slice(0, w)) if s == "L" else (slice(None), slice(W - w, W)) if s == "R" else (slice(0, w), slice(None)) if s == "T" else (slice(H - w, H), slice(None))
                a, b = old[sl], new[sl]
                hl = a.copy()
                hl[diff[sl]] = (255, 0, 0)
                f = min(1.0, 900 / max(a.shape[:2]))
                tiles = [cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if k < 2 else cv2.INTER_NEAREST) for k, t in enumerate((a, b, hl))]
                sep = np.full((tiles[0].shape[0], 4, 3), (0, 0, 255), np.uint8) if s in "LR" else np.full((4, tiles[0].shape[1], 3), (0, 0, 255), np.uint8)
                parts = [tiles[0], sep, tiles[1], sep, tiles[2]]
                Path(outdir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(outdir) / f"f{fid:04d}_p{i}.png"), cv2.cvtColor(np.hstack(parts) if s in "LR" else np.vstack(parts), cv2.COLOR_RGB2BGR))
            out.append(rec)
        except Exception as e:  # noqa
            out.append(dict(fid=fid, page=i, err=repr(e)))
        finally:
            geometry.BINDING_HOLES = True
    return out


def main():
    src, name = Path(sys.argv[1]), sys.argv[2]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 16
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in EXT_IMG | {".pdf"} and not p.name.startswith("contact_"))
    outdir = HERE / "work" / f"changed_{name}"
    t0 = time.time()
    recs = []
    with ProcessPoolExecutor(procs) as ex:
        for r in ex.map(work, [(i, str(p), str(outdir)) for i, p in enumerate(files)], chunksize=1):
            recs.extend(r)
    ok = [r for r in recs if "err" not in r]
    fired = [r for r in ok if r["fired"]]
    summ = dict(set=name, files=len(files), pages=len(ok), unreadable_pages=len(recs) - len(ok), fired_pages=len(fired),
                fired_without_pixel_change=sum(1 for r in fired if not r["changed"]),
                changed_without_firing=sum(1 for r in ok if r["changed"] and not r["fired"]),
                max_changed_frac=max((r["changed_frac"] for r in fired), default=0.0),
                fired=fired, seconds=round(time.time() - t0, 1))
    (HERE / f"eval_{name}.json").write_text(json.dumps(summ, indent=1))
    (HERE / "work" / f"files_{name}.json").write_text(json.dumps({i: str(p) for i, p in enumerate(files)}))
    print({k: v for k, v in summ.items() if k != "fired"})


if __name__ == "__main__":
    main()
