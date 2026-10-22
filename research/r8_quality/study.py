"""R8 quality study: encode cleaned pages with candidate settings, render, measure size and fidelity.

    python study.py SET OUT.jsonl [--ocr] [--limit N]

SET is a name in CANDIDATES (see below).  One row per (page, candidate).  Images/PDFs live in work/ (git-ignored).
Pages: the 26 native-300 ppi pages of the private corpus (research/corpus) plus up to 3 pages of each of the sample PDFs in
work/samples (copied from the private samples/ tree).  All pages are cleaned once with the shipped pipeline (flatten, no resize).
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pikepdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from scanpp import encode, pdfio  # noqa: E402
from scanpp.encode import ImageSpec, QualitySettings  # noqa: E402
from scanpp.options import Options  # noqa: E402

CORPUS = Path(os.environ.get("CORPUS", ROOT / "research" / "corpus"))
WORK = HERE / "work"
QS = QualitySettings


# ----------------------------------------------------------------------------- candidates
def _flate(rgb, dpi, size, jpeg_dpi=None, qs=None):
    """Lossless Flate page (PNG predictor), gray when the page has no colour."""
    gray = not encode.has_color(rgb)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray else rgb
    ok, png = cv2.imencode(".png", img if gray else cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    b = png.tobytes()
    idat, pos = [], 8
    while pos < len(b):
        n = int.from_bytes(b[pos:pos + 4], "big")
        if b[pos + 4:pos + 8] == b"IDAT":
            idat.append(b[pos + 8:pos + 8 + n])
        pos += 12 + n
    h, w = img.shape[:2]
    spec = ImageSpec(b"".join(idat), w, h, "DeviceGray" if gray else "DeviceRGB", 8, "FlateDecode",
                     {"DecodeParms": {"Predictor": 15, "Colors": 1 if gray else 3, "BitsPerComponent": 8, "Columns": w}})
    return encode.EncodedPage({"Im0": spec}, encode._full_page_content(*size), info={"scheme": "flate", "gray": gray})


def _jpeg(ppi, q):
    return lambda rgb, dpi, size, jd=None, qs=None: encode._encode_e1(rgb, dpi, size, ppi, "e1", q)


def _mask(col_ppi, col_q, mask_ppi=300):
    q = QS(col_ppi=col_ppi, col_q=col_q, mask_ppi=mask_ppi)
    return lambda rgb, dpi, size, jd=None, qs=None: encode.encode_auto(rgb, dpi, size, None, q)


def _auto(qs):
    return lambda rgb, dpi, size, jd=None, _q=None: encode.encode_auto(rgb, dpi, size, None, qs)


def _level(n):
    return lambda rgb, dpi, size, jd=None, qs=None: encode.encode_page(rgb, dpi, size, "auto", None, n)


def candidates(name: str) -> dict:
    c: dict = {}
    if name == "sweep":
        # auto-routed families (E1 route for highlighter pages stays at 150/q55), masks with colour layer settings
        for p, q in [(75, 25), (75, 40), (100, 25), (100, 40), (100, 55), (125, 40), (125, 55), (150, 40), (150, 55),
                     (150, 70), (200, 55), (200, 70)]:
            c[f"mask_c{p}_q{q}"] = _auto(QS(col_ppi=p, col_q=q))
        c["mask_m200_c100_q40"] = _auto(QS(mask_ppi=200))
        # E1 whole-page JPEG families
        for p, q in [(150, 25), (150, 35), (150, 45), (150, 55), (150, 70), (200, 40), (200, 55), (200, 70), (200, 80),
                     (250, 60), (250, 75), (250, 85), (300, 60), (300, 75), (300, 85), (300, 90), (300, 93), (300, 96)]:
            c[f"jpeg_p{p}_q{q}"] = _jpeg(p, q)
        c["flate_300"] = _flate
    elif name == "sweep2":
        for p, q in [(75, 32), (100, 32), (125, 48)]:
            c[f"mask_c{p}_q{q}"] = _auto(QS(col_ppi=p, col_q=q))
        for p, q in [(150, 40), (150, 50), (150, 62), (200, 65), (250, 65), (250, 72), (300, 40), (300, 50), (300, 82), (300, 92)]:
            c[f"jpeg_p{p}_q{q}"] = _jpeg(p, q)
    elif name == "ladder":
        for n in range(1, 11):
            c[f"L{n}"] = _level(n)
    elif name == "ocr_dev":
        for n in (1, 5, 10):
            c[f"L{n}"] = _level(n)
    else:
        raise SystemExit(f"unknown set {name}")
    return c


# ----------------------------------------------------------------------------- pages
def page_list(limit: int | None = None):
    pages = []
    man = json.loads((CORPUS / "manifest.json").read_text())
    for m in man:
        if m["dpi"][0] >= 290 or m["file"].startswith("single26"):
            pages.append(("c", m["file"], "corpus:" + m["file"].rsplit(".", 1)[0]))
    for f in sorted((WORK / "samples").glob("s*.pdf")):
        with pikepdf.open(f) as pdf:
            n = len(pdf.pages)
        for i in range(min(n, 3)):
            pages.append(("s", f"{f.name}:{i}", f"{f.stem}:{i + 1}"))
    return pages[:limit] if limit else pages


def load_page(kind: str, ref: str) -> np.ndarray:
    if kind == "c":
        a = cv2.imread(str(CORPUS / ref), cv2.IMREAD_COLOR)
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    name, i = ref.split(":")
    with pikepdf.open(WORK / "samples" / name) as pdf:
        img = next(iter(pdf.pages[int(i)].resources.XObject.values()))
        a = cv2.imdecode(np.frombuffer(img.read_raw_bytes(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- worker
def run_page(args):
    kind, ref, pid, setname, do_ocr = args
    import metrics
    cv2.setNumThreads(1)
    from scanpp import pipeline
    t0 = time.time()
    rgb = load_page(kind, ref)
    out, dpi, stats = pipeline.process_array(rgb, 300.0, Options(), allow_drop=False, resize=False)
    if out is None or encode.select_scheme(out, dpi) == "blank":
        return []
    h, w = out.shape[:2]
    size = (w * 72 / 300.0, h * 72 / 300.0)
    scheme5 = encode.select_scheme(out, dpi)
    tmp = WORK / "tmp"
    tmp.mkdir(exist_ok=True)
    rows = []
    ref_words = None
    if do_ocr:
        import ocrlib
        ref_words = ocrlib.words(out)
    for cname, fn in candidates(setname).items():
        t1 = time.time()
        ep = fn(out, dpi, size, None, None)
        t_enc = time.time() - t1
        ep.media_box = (0.0, 0.0, size[0], size[1])
        path = tmp / f"{os.getpid()}.pdf"
        pdfio.write_pdf([ep], path)
        pdf_bytes = path.stat().st_size
        ren = recon(ep, out, dpi)
        m = metrics.metrics(out, ren)
        if do_ocr and ref_words and len(ref_words) >= 20:
            m["ocr_recall"] = ocrlib.recall(ref_words, ocrlib.hyp_words(metrics._fit(out, metrics.render_pdf(path, 300))))
        with pikepdf.open(path) as pdf:
            filters = sorted({str(v.get("/Filter")) for v in _images(pdf)})
        rows.append({"page": pid, "scheme5": scheme5, "cand": cname, "bytes": ep.nbytes, "pdf_bytes": pdf_bytes,
                     "enc_s": round(t_enc, 3), "scheme": ep.info.get("scheme"), "filters": filters, **m})
    path.unlink(missing_ok=True)
    return rows


def _decode_spec(sp: ImageSpec) -> np.ndarray:
    """Pixels of a DCT / Flate image spec as RGB (gray replicated)."""
    if sp.filter == "FlateDecode":
        import struct
        ch = 1 if sp.colorspace == "DeviceGray" else 3

        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", sp.width, sp.height, 8, 0 if ch == 1 else 2, 0, 0, 0))
               + chunk(b"IDAT", sp.data) + chunk(b"IEND", b""))
        a = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    else:
        a = cv2.imdecode(np.frombuffer(sp.data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def recon(ep, ref: np.ndarray, dpi: float) -> np.ndarray:
    """What the page decodes to, on the reference grid, independent of any PDF renderer: the page image is decoded and bilinearly
    upsampled to the reference size; for mask schemes the (lossless) ink mask is recomputed and the colour layer shows through it."""
    from scanpp import mrc
    h, w = ref.shape[:2]
    sp = next(iter(ep.xobjects.values()))
    img = _decode_spec(sp)
    if sp.mask is None:
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img
    r, ppi = encode._to_mask_ppi(ref, dpi, ep.info["mask_ppi"])
    ink = mrc.ink_mask(r, ppi)
    mh, mw = ink.shape
    up = cv2.resize(img, (mw, mh), interpolation=cv2.INTER_LINEAR)
    page = np.where(ink[..., None], up, 255).astype(np.uint8)
    if (mh, mw) != (h, w):
        page = cv2.resize(page, (w, h), interpolation=cv2.INTER_LINEAR)
    return page


def _images(pdf):
    seen = []

    def walk(o):
        if o.get("/Subtype") == "/Image":
            seen.append(o)
            for k in ("/Mask", "/SMask"):
                if k in o:
                    walk(o[k])
    for pg in pdf.pages:
        for v in pg.resources.get("/XObject", {}).values():
            walk(v)
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set")
    ap.add_argument("out")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    os.environ["OMP_THREAD_LIMIT"] = "1"
    cv2.setNumThreads(1)
    pages = page_list(a.limit)
    tasks = [(k, r, pid, a.set, a.ocr) for k, r, pid in pages]
    n = 0
    with ProcessPoolExecutor(a.jobs) as ex, open(a.out, "w") as f:
        for rows in ex.map(run_page, tasks):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            n += 1
            print(f"\r{n}/{len(tasks)}", end="", file=sys.stderr)
    print(file=sys.stderr)


if __name__ == "__main__":
    main()
