"""Content-loss detector: words OCR'd (tesseract, R1 settings, conf >= 80) on the ORIGINAL page that are missing from the
OCR of (a) the plain cleaned page (no encoding; clean stage) and (b) the final rendered output (pdftoppm 300 dpi; total).
Output words are taken from psm 3 and psm 11 (tesseract's layout analysis flips on whole pages; a word is lost only if neither finds it).
Fuzzy matching: exact (case-insensitive) or edit distance <= 1 for words >= 4 chars, within MAXD px (300 dpi) of the position.

Usage: lostwords.py --src SRC_DIR --out OUT.json [--n 150] [--seed 3] [--extra] [--procs 12] [--pages file:page,...]
  --src   directory containing the `scanpp` package to test (default: repo src)
  --extra also run the Ace stations p6 and the Epson keymap (psm 11)
Pages: random (seed) sample of single-image scan-PDF pages of samples/ (copied to a scratch dir first, never written)."""
import os, sys, json, argparse, random, shutil, subprocess, tempfile, csv, io
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=os.path.join(ROOT, "src")); ap.add_argument("--out", required=True)
ap.add_argument("--n", type=int, default=150); ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--procs", type=int, default=12); ap.add_argument("--extra", action="store_true")
ap.add_argument("--list", help="JSON list of [pdf, page] to use instead of random sampling")
ap.add_argument("--scratch", default=os.environ.get("TMPDIR", "/tmp"))
A = ap.parse_args()
sys.path.insert(0, os.path.abspath(A.src))
import numpy as np, cv2
cv2.setNumThreads(1)
from PIL import Image
import re

MINCONF, MAXD = 80, 80


def norm(t):
    t = re.sub(r"^[^\w]+|[^\w]+$", "", t.strip())
    return t.lower() if re.search(r"[A-Za-z0-9]", t) else None


def tess(gray, psm, tmp):
    fd, p = tempfile.mkstemp(suffix=".png", dir=tmp); os.close(fd)
    cv2.imwrite(p, gray)
    out = subprocess.run(["tesseract", p, "stdout", "-l", "eng", "--psm", str(psm), "--oem", "1", "--dpi", "300", "tsv"],
                         capture_output=True, text=True, env=dict(os.environ, OMP_THREAD_LIMIT="1")).stdout
    os.remove(p)
    rows = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        if r["level"] != "5":
            continue
        t = norm(r["text"] or "")
        if t:
            rows.append(dict(text=t, conf=float(r["conf"]), cx=int(r["left"]) + int(r["width"]) / 2, cy=int(r["top"]) + int(r["height"]) / 2,
                             x=int(r["left"]), y=int(r["top"]), w=int(r["width"]), h=int(r["height"])))
    return rows


def lev(a, b):
    if abs(len(a) - len(b)) > 1:
        return 2
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lost_words(ref, hyp):
    """ref words (conf >= MINCONF) with no fuzzy+positional match among hyp words (any confidence)."""
    lost = []
    for r in ref:
        if r["conf"] < MINCONF or len(r["text"]) < 2:
            continue
        ok = False
        for h in hyp:
            if abs(h["cx"] - r["cx"]) > MAXD or abs(h["cy"] - r["cy"]) > MAXD:
                continue
            if h["text"] == r["text"] or (len(r["text"]) >= 4 and len(h["text"]) >= 3 and lev(h["text"], r["text"]) <= 1):
                ok = True; break
        if not ok:
            lost.append(r)
    return lost


def gray300(rgb, dpi):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if abs(dpi - 300) > 1:
        s = 300.0 / dpi
        g = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return g


def enhance(g):
    """Flat-field the gray page and amplify its ink 3x: faint printed text that tesseract misses on the raw scan becomes readable,
    so it can serve as ground truth ("this word was on the page")."""
    bg = cv2.GaussianBlur(cv2.dilate(g, np.ones((15, 15), np.uint8)), (0, 0), 8).astype(np.float32)
    d = np.clip(1.0 - g.astype(np.float32) / np.maximum(bg, 1), 0, 1)
    return np.clip(255 * (1 - 3.0 * d), 0, 255).astype(np.uint8)


def merge_ref(raw, enh):
    out = list(raw)
    for e in enh:
        if not any(abs(e["cx"] - r["cx"]) < max(r["w"], e["w"]) / 2 + 5 and abs(e["cy"] - r["cy"]) < max(r["h"], e["h"]) / 2 + 5 for r in raw):
            out.append(e)
    return out


def job(item):
    import pikepdf
    from scanpp import pdfio, pipeline
    from scanpp.options import Options
    name, src, pno, psm = item["name"], item["path"], item["page"], item.get("psm", 3)
    tmp = tempfile.mkdtemp(dir=A.scratch)
    try:
        if src.lower().endswith(".pdf"):
            with pikepdf.open(src) as pdf:
                t = pdfio.extract_page_input(pdf.pages[pno - 1], pno - 1)
            rgb = pipeline._decode(t); dpi = t.dpi; rot = t.rotate; mb = t.media_box
            if rot:
                rgb = np.ascontiguousarray(np.rot90(rgb, -rot // 90))
        else:
            rgb = np.asarray(Image.open(src).convert("RGB")).copy(); dpi = 300.0; rot = 0
        up = rgb.copy()
        opts = Options()
        out, od, st = pipeline.process_array(rgb.copy(), dpi, opts, allow_drop=False, resize=False)
        if out is None:
            return dict(name=name, skipped="dropped")
        g0 = gray300(up, dpi)
        ref = merge_ref(tess(g0, psm, tmp), tess(enhance(g0), psm, tmp))
        gc = gray300(out, od)
        cl = tess(gc, psm, tmp) + (tess(gc, 11, tmp) if psm != 11 else [])      # a word counts as present if either layout mode finds it
        # encode + render
        from scanpp.encode import encode_page
        o2 = np.ascontiguousarray(np.rot90(out, rot // 90)) if rot else out
        h, w = o2.shape[:2]
        bw, bh = ((mb[2] - mb[0], mb[3] - mb[1]) if src.lower().endswith(".pdf") else (w * 72 / od, h * 72 / od))
        ep = encode_page(o2, od, (bw, bh), "auto", None)
        ep.media_box = mb if src.lower().endswith(".pdf") else (0.0, 0.0, bw, bh); ep.rotate = rot
        pdfio.write_pdf([ep], tmp + "/o.pdf")
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", "-gray", tmp + "/o.pdf", tmp + "/r"], check=True)
        rg = cv2.imread(tmp + "/r.png", cv2.IMREAD_GRAYSCALE)
        if rot:
            rg = np.ascontiguousarray(np.rot90(rg, -rot // 90))
        enc = tess(rg, psm, tmp) + (tess(rg, 11, tmp) if psm != 11 else [])
        nref = sum(1 for r in ref if r["conf"] >= MINCONF and len(r["text"]) >= 2)
        lc, le = lost_words(ref, cl), lost_words(ref, enc)
        return dict(name=name, scheme=ep.info.get("scheme"), nref=nref, bytes=ep.nbytes,
                    lost_clean=[[w_["text"], w_["x"], w_["y"], w_["w"], w_["h"], w_["conf"]] for w_ in lc],
                    lost_total=[[w_["text"], w_["x"], w_["y"], w_["w"], w_["h"], w_["conf"]] for w_ in le], dpi=od)
    except Exception as e:  # noqa: BLE001
        import traceback
        return dict(name=name, error=repr(e) + traceback.format_exc()[-300:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sample_pages(n, seed):
    import pikepdf
    from scanpp import pdfio
    pdfs = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(ROOT, "samples")) for f in fs if f.lower().endswith(".pdf"))
    pages = []
    for p in pdfs:
        try:
            with pikepdf.open(p) as pdf:
                ok, _ = pdfio.classify_pdf(pdf)
                if ok:
                    pages += [(p, i + 1) for i in range(len(pdf.pages))]
        except Exception:  # noqa: BLE001
            pass
    random.Random(seed).shuffle(pages)
    return pages[:n]


if __name__ == "__main__":
    os.makedirs(A.scratch, exist_ok=True)
    items = []
    if A.list:
        pgs = json.load(open(A.list))
    else:
        pgs = sample_pages(A.n, A.seed)
    for p, i in pgs:
        items.append(dict(name=f"{os.path.relpath(p, os.path.join(ROOT, 'samples'))}#{i}", path=p, page=i))
    if A.extra:
        ace = os.path.join(ROOT, "samples/2026-2027/Sec 2/Review - The Case of Ace - Stations.pdf")
        items.append(dict(name="ACE_stations#6", path=ace, page=6))
        items.append(dict(name="EPSON_keymap", path=os.path.join(ROOT, "research/scans/epson_300dpi.png"), page=1, psm=11))
    from multiprocessing import Pool
    with Pool(A.procs) as pool:
        res = pool.map(job, items, chunksize=1)
    json.dump(res, open(A.out, "w"), indent=1)
    ok = [r for r in res if "nref" in r]
    tn = sum(r["nref"] for r in ok); lc = sum(len(r["lost_clean"]) for r in ok); lt = sum(len(r["lost_total"]) for r in ok)
    print(f"pages {len(ok)}/{len(res)}  ref words {tn}  lost(clean) {lc} = {lc / max(tn, 1) * 100:.3f}%   lost(total) {lt} = {lt / max(tn, 1) * 100:.3f}%"
          f"   bytes/page {np.mean([r['bytes'] for r in ok]) / 1024:.1f} KB")
    for r in sorted(ok, key=lambda r: -len(r["lost_total"]))[:15]:
        print(f"  {len(r['lost_total']):3d}/{r['nref']:4d} total  {len(r['lost_clean']):3d} clean  {r['scheme']}  {r['name']}")
    for r in res:
        if "error" in r:
            print("ERR", r["name"], r["error"][:200])
