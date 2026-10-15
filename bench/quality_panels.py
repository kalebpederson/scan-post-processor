"""Before/after panels for the content-quality fixes: bench/panels/quality_*.png.
Usage: quality_panels.py --base path/to/old_clean.py   (new = src/scanpp/clean.py).  Rows: original | base | new, each rendered through
clean_page -> encode_page (auto) -> pdftoppm at the page's native ppi; crops are 1:1 (x2 for display)."""
import sys, os, argparse, subprocess, tempfile, importlib.util
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, cv2, pikepdf
from PIL import Image, ImageDraw
from scanpp import pdfio, pipeline, encode
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--out", default=os.path.join(ROOT, "bench", "panels")); A = ap.parse_args()
S = os.path.join(ROOT, "samples")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


base, new = load(A.base, "base_clean"), load(os.path.join(ROOT, "src/scanpp/clean.py"), "new_clean")


def page(pdf, n):
    with pikepdf.open(os.path.join(S, pdf)) as p:
        t = pdfio.extract_page_input(p.pages[n - 1], n - 1)
    return pipeline._decode(t), t.dpi, t.media_box


def through(clean_mod, rgb, dpi, mb):
    out = clean_mod.clean_page(rgb, dpi)
    ep = encode.encode_page(out, dpi, (mb[2] - mb[0], mb[3] - mb[1]), "auto"); ep.media_box = mb
    with tempfile.TemporaryDirectory() as td:
        pdfio.write_pdf([ep], td + "/o.pdf")
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", "-singlefile", td + "/o.pdf", td + "/r"], check=True)
        r = np.asarray(Image.open(td + "/r.png").convert("RGB"))
    return r, ep.info.get("scheme")


def panel(name, pdf, n, box, title, zoom=2):
    rgb, dpi, mb = page(pdf, n)
    y0, y1, x0, x1 = box
    rb, sb = through(base, rgb, dpi, mb); rn, sn = through(new, rgb, dpi, mb)
    rows = []
    for lab, im in (("original", rgb), (f"before ({sb})", rb), (f"after ({sn})", rn)):
        c = im[y0:y1, x0:x1]
        c = cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST if zoom > 1 else cv2.INTER_AREA)
        pil = Image.fromarray(c); d = ImageDraw.Draw(pil); d.rectangle((0, 0, 230, 16), fill=(255, 255, 200)); d.text((3, 2), lab, fill=(0, 0, 0))
        rows.append(np.asarray(pil))
    img = np.vstack(rows)
    Image.fromarray(img).save(os.path.join(A.out, f"quality_{name}.png")); print(name, title, img.shape)


ACE = "2026-2027/Sec 2/Review - The Case of Ace - Stations.pdf"
panel("ace_footer", ACE, 6, (1850, 1910, 140, 1600), "Defect A: faint gray footer", zoom=1)
panel("pencil_yellow", "2026-2027/Sec 2/Data/Sec 2 B8 Formative 2.pdf", 43, (1750, 2000, 100, 700), "Defect B(i): pencil on yellow")
panel("blue_fringe", "2022-2023/Secondary 2 Honors/9.99 Project - Art with Transformations.pdf", 4, (2680, 2900, 200, 800), "Defect B(ii): fringes on blue paper")
panel("noreg_pens", ACE, 6, (1050, 1300, 100, 900), "no regression: coloured pens + table", zoom=1)
panel("noreg_text", "2026-2027/Sec 2/Data/Sec 2 B8 Formative 2.pdf", 43, (180, 420, 60, 860), "no regression: printed text + pencil on yellow", zoom=1)
