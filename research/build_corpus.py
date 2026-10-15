#!/usr/bin/env python3
"""Build research/corpus from samples/ (seed 42). Lossless extraction of embedded page images."""
import json, random, shutil, collections
from pathlib import Path
import pikepdf
from pikepdf import PdfImage
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CORP = ROOT / "research" / "corpus"
rng = random.Random(42)

def extract(pdf_path, page_no, dest_stem):
    """Write the page's embedded image. DCT -> raw JPEG bytes copied; otherwise lossless PNG."""
    with pikepdf.open(pdf_path) as pdf:
        page = pdf.pages[page_no - 1]
        imgs = list(page.images.items())
        name, obj = max(imgs, key=lambda kv: int(kv[1].Width) * int(kv[1].Height))
        filt = obj.get("/Filter")
        filters = [str(f) for f in filt] if isinstance(filt, pikepdf.Array) else ([str(filt)] if filt is not None else [])
        if filters == ["/DCTDecode"]:
            out = Path(str(dest_stem) + ".jpg")
            out.write_bytes(obj.read_raw_bytes())
            enc = "jpeg-copied"
        else:
            out = Path(str(dest_stem) + ".png")
            pi = PdfImage(obj)
            pil = pi.as_pil_image()
            pil.save(out)
            enc = "+".join(filters) or "raw"
    return out, enc

def main():
    if CORP.exists():
        shutil.rmtree(CORP)
    CORP.mkdir(parents=True)
    survey = json.loads((ROOT / "research" / "survey.json").read_text())
    scans = [r for r in survey["pdfs"] if r["class"] == "scan" and r["images"]]
    scans.sort(key=lambda r: r["path"])
    year = lambda r: r["path"].split("/")[1]
    manifest = []

    # pairs: Simple Scan, even page count 2..8, jpeg, all pages same ppi/dims -> duplex-looking
    def duplexish(r):
        if not r["simple_scan_tag"] or r["pages"] % 2 or not (2 <= r["pages"] <= 8): return False
        im = [i for i in r["images"] if i["type"] == "image"]
        return len(im) == r["pages"] and all(i["enc"] == "jpeg" for i in im) and len({(i["ppi"][0], i["w"]) for i in im}) == 1
    dup = [r for r in scans if duplexish(r)]
    by_year = collections.defaultdict(list)
    for r in dup: by_year[year(r)].append(r)
    pair_docs = []
    ys = sorted(by_year)
    for y in ys: rng.shuffle(by_year[y])
    while len(pair_docs) < 10 and any(by_year.values()):
        for y in ys:
            if by_year[y] and len(pair_docs) < 10: pair_docs.append(by_year[y].pop())
    used_docs = set()
    pid = 0
    for r in pair_docs:
        k = rng.randrange(r["pages"] // 2)
        p1 = 2 * k + 1
        pid += 1
        for side, pg in (("front", p1), ("back", p1 + 1)):
            stem = CORP / f"pair{pid:02d}_{side}"
            out, enc = extract(ROOT / r["path"], pg, stem)
            manifest.append(entry(out, r, pg, enc, f"pair{pid:02d}", side))
        used_docs.add(r["path"])

    # singles: round-robin across years, <=2 per doc
    pool = [r for r in scans if r["path"] not in used_docs]
    by_year = collections.defaultdict(list)
    for r in pool: by_year[year(r)].append(r)
    for y in by_year: rng.shuffle(by_year[y])
    picks = []; cnt = collections.Counter()
    while len(picks) < 40 and any(by_year.values()):
        for y in sorted(by_year):
            if by_year[y] and len(picks) < 40:
                r = by_year[y].pop()
                n = min(2, r["pages"])
                for pg in rng.sample(range(1, r["pages"] + 1), 1 if len(picks) % 3 else n):
                    if len(picks) < 40: picks.append((r, pg))
    for i, (r, pg) in enumerate(picks):
        out, enc = extract(ROOT / r["path"], pg, CORP / f"single{i:02d}")
        manifest.append(entry(out, r, pg, enc, None, "single"))

    # raster samples: document scans first, then some photo-ish at least 1000px
    imgs = [r for r in survey["images"] if r["kind"] == "document_scan" and r["w"] >= 1000 and min(r["w"], r["h"]) >= 1000]
    imgs.sort(key=lambda r: r["path"])
    names = collections.defaultdict(list)
    for r in imgs: names[Path(r["path"]).parent.as_posix()].append(r)
    chosen = []
    dirs = sorted(names); rng.shuffle(dirs)
    for d in dirs:
        if len(chosen) < 6: chosen.append(rng.choice(names[d]))
    if len(chosen) < 6:
        rest = [r for r in imgs if r not in chosen]; rng.shuffle(rest); chosen += rest[:6 - len(chosen)]
    for i, r in enumerate(chosen):
        src = ROOT / r["path"]
        dst = CORP / f"image{i:02d}{src.suffix.lower()}"
        shutil.copyfile(src, dst)
        manifest.append({"file": dst.name, "source": r["path"], "page": None, "pair_id": None, "side": "image",
                         "dpi": r["dpi"], "dims": [r["w"], r["h"]], "orig_encoding": r["format"].lower(), "bytes": dst.stat().st_size})
    (CORP / "manifest.json").write_text(json.dumps(manifest, indent=1))
    contact(manifest)
    print(len(manifest), "corpus files;", collections.Counter(m["side"] for m in manifest))
    print("years:", dict(collections.Counter(m["source"].split("/")[1] for m in manifest)))

def entry(out, r, pg, enc, pair, side):
    im = [i for i in r["images"] if i["page"] == pg and i["type"] == "image"][0]
    return {"file": out.name, "source": r["path"], "page": pg, "pair_id": pair, "side": side, "dpi": im["ppi"],
            "dims": [im["w"], im["h"]], "orig_encoding": im["enc"], "extraction": enc, "bytes": out.stat().st_size,
            "pdf_page_size": r["page_size"]}

def contact(manifest, cols=8, tw=240, per_sheet=24):
    # 4x-reduced thumbnails (relative to a ~1000px-wide... use fixed width), max 2000 px wide
    from PIL import ImageDraw
    files = [m["file"] for m in manifest]
    for s in range(0, len(files), per_sheet):
        chunk = files[s:s + per_sheet]; cols_ = 8; th = 310
        rows = (len(chunk) + cols_ - 1) // cols_
        sheet = Image.new("RGB", (cols_ * tw, rows * (th + 14)), "gray")
        d = ImageDraw.Draw(sheet)
        for i, f in enumerate(chunk):
            im = Image.open(CORP / f).convert("RGB")
            im.thumbnail((tw - 4, th)); 
            x, y = (i % cols_) * tw + 2, (i // cols_) * (th + 14)
            sheet.paste(im, (x, y + 12)); d.text((x, y), f[:34], fill="white")
        sheet.save(CORP / f"contact_{s // per_sheet + 1}.png")

main()
