"""Which pages do the r7 flatbed edge rules change?  Edge-clean each page twice in one process: with the r7 rules switched off
(`boundary.EDGE_LINE`, `geometry.FLATBED_COIL_FRINGE`, `geometry.FLATBED_CORNER_DOTS`, `geometry.FLATBED_HAIRLINE` = False, i.e. the previous behaviour) and with them on.

usage: eval_edges.py SRC_DIR NAME [--procs 16]
Writes research/r7_flatbed/eval_NAME.json (aggregate; pages keyed by an index, never by file name) and, local only,
research/r7_flatbed/work/files_NAME.json (index -> path) plus before/after crops of every changed page in
research/r7_flatbed/work/changed_NAME/ (left: previous output, middle: new output, right: previous with the newly painted pixels in red, around the changed pixels).
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pikepdf

from scanpp import boundary, geometry, imageio, pdfio, pipeline

HERE = Path(__file__).resolve().parent
EXT_IMG = {".jpg", ".jpeg", ".png"}


def pages_of(path):
    ext = path.suffix.lower()
    if ext in EXT_IMG:
        try:
            rgb, dpi, _ = imageio.read_image(path)
        except Exception as e:  # noqa
            yield 0, None, 0, repr(e)
            return
        yield 0, rgb, dpi, None
    elif ext == ".pdf":
        try:
            pdf = pikepdf.open(path)
        except Exception as e:  # noqa
            yield 0, None, 0, repr(e)
            return
        with pdf:
            for i in range(len(pdf.pages)):
                try:
                    t = pdfio.extract_page_input(pdf.pages[i], i, path.name)
                    rgb = pipeline._decode(t)
                    if t.rotate:
                        rgb = np.ascontiguousarray(np.rot90(rgb, -t.rotate // 90))
                    yield i, rgb, t.dpi, None
                except Exception as e:  # noqa
                    yield i, None, 0, repr(e)


def set_rules(on: bool):
    boundary.EDGE_LINE = on
    geometry.FLATBED_COIL_FRINGE = on
    geometry.FLATBED_CORNER_DOTS = on
    geometry.FLATBED_HAIRLINE = on


def work(args):
    fid, path, outdir = args
    out = []
    for i, rgb, dpi, err in pages_of(Path(path)):
        if rgb is None:
            out.append(dict(fid=fid, page=i, err=err))
            continue
        try:
            set_rules(False)
            old, _ = geometry.clean_edges(rgb, dpi)
            set_rules(True)
            new, info = geometry.clean_edges(rgb, dpi)
            g_, d_ = geometry.prep_gray(rgb, dpi)
            m_, i_, _ = geometry._edge_mask_work_g(g_, d_)
            new_line_sides = {r['side'] for r in geometry._paper_boundaries(rgb, g_, d_, i_) if r.get('kind') == 'line'}
            diff = (old != new).any(axis=2)
            rec = dict(fid=fid, page=i, dpi=round(float(dpi), 1), changed=bool(diff.any()),
                       line=sorted(k[-1] for k in info if k.startswith("boundary_") and k[-1] in new_line_sides),
                       fringe=int(info.get("fringe_px", 0)), corner=int(info.get("corner_dots", 0)),
                       hairline=sorted(k[-1] for k in info if k.startswith("hairline_")))
            if rec["changed"]:
                rec["changed_frac"] = float(diff.mean())
                ys, xs = np.nonzero(diff)
                H, W = diff.shape
                pad = int(0.25 * dpi)
                y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad)
                x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
                # a changed region that spans the whole border is shown as a 1.2 in strip, not the full page
                y1 = min(y1, y0 + int(4 * dpi))
                x1 = min(x1, x0 + int(4 * dpi))
                a, b = old[y0:y1, x0:x1], new[y0:y1, x0:x1]
                f = min(1.0, 600 / max(a.shape[:2]))
                a = cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                hl = old[y0:y1, x0:x1].copy()
                hl[diff[y0:y1, x0:x1]] = (255, 0, 0)               # pixels the new rules painted, in red on the old output
                hl = cv2.resize(hl, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
                sep = np.full((a.shape[0], 4, 3), (0, 0, 255), np.uint8)
                Path(outdir).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(outdir) / f"f{fid:04d}_p{i}.png"), cv2.cvtColor(np.hstack([a, sep, b, sep, hl]), cv2.COLOR_RGB2BGR))
            out.append(rec)
        except Exception as e:  # noqa
            out.append(dict(fid=fid, page=i, err=repr(e)))
        finally:
            set_rules(True)
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
    ch = [r for r in ok if r["changed"]]
    summ = dict(set=name, files=len(files), pages=len(ok), unreadable_pages=len(recs) - len(ok),
                changed_pages=len(ch),
                by_rule=dict(line=sum(1 for r in ch if r["line"]), coil_fringe=sum(1 for r in ch if r["fringe"]), corner_dots=sum(1 for r in ch if r["corner"]),
                             hairline=sum(1 for r in ch if r["hairline"])),
                max_changed_frac=max((r["changed_frac"] for r in ch), default=0.0),
                changed=[{k: v for k, v in r.items()} for r in ch], seconds=round(time.time() - t0, 1))
    (HERE / f"eval_{name}.json").write_text(json.dumps(summ, indent=1))
    (HERE / "work" / f"files_{name}.json").write_text(json.dumps({i: str(p) for i, p in enumerate(files)}))
    print({k: v for k, v in summ.items() if k != "changed"})


if __name__ == "__main__":
    main()
