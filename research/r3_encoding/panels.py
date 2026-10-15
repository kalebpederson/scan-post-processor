"""Viewer-style comparison panels. Each row = candidate; columns = 110 ppi laptop @100%, 110 ppi @200% (=220 ppi), 160 ppi tablet @100%.
Candidates are PDFs rendered with pdfium (Chrome engine) AT those ppi (so the viewer's own resampling is included);
the cleaned reference is shown as a raster area-resampled to the same ppi (what an ideal lossless viewer would show).
usage: panels.py name page x0,y0,x1,y1 cfg1,cfg2,... [title]   (box in 300-ppi page pixels)"""
import sys
from config import *
import render
import pypdfium2 as pdfium_
VIEWS = [("110ppi @100%", 110), ("110ppi @200%", 220), ("160ppi @100%", 160)]

def label(img, text, h=18):
    bar = np.full((h, img.shape[1], 3), 255, np.uint8); cv2.putText(bar, text, (3, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160), 1, cv2.LINE_AA)
    return np.vstack([bar, img])

def viewer_crop(pdf, box, ppi):
    d = pdfium_.PdfDocument(pdf); a = np.asarray(d[0].render(scale=ppi / 72.0, rev_byteorder=True).to_numpy())[:, :, :3]; d.close()
    f = ppi / 300.0; x0, y0, x1, y1 = box
    return a[round(y0 * f):round(y1 * f), round(x0 * f):round(x1 * f)].copy()
def ref_crop(ref, box, ppi):
    f = ppi / 300.0; x0, y0, x1, y1 = box
    small = cv2.resize(ref, (round(ref.shape[1] * f), round(ref.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return small[round(y0 * f):round(y1 * f), round(x0 * f):round(x1 * f)].copy()

def panel(name, page, box, cfgs, title=None, sizes=None):
    ref = load_clean(page); rows = []
    for c in ["REF"] + cfgs:
        cells = []
        for vn, ppi in VIEWS:
            if c == "REF": cr = ref_crop(ref, box, ppi); lab = "cleaned reference (300 ppi) | " + vn
            else:
                cr = viewer_crop(str(WORK / "pdf" / c / f"{page}.pdf"), box, ppi)
                lab = f"{c} {sizes.get(c, '') if sizes else ''} | {vn}"
            cells.append(label(cr, lab))
        H = max(x.shape[0] for x in cells)
        cells = [np.pad(x, ((0, H - x.shape[0]), (0, 8), (0, 255 - 255)), constant_values=255) for x in cells]
        rows.append(np.hstack(cells))
    W = max(r.shape[1] for r in rows)
    img = np.vstack([np.pad(r, ((0, 5), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
    img = label(img, (title or name) + f"  [{page} box {box}]", h=22)
    if img.shape[1] > 2000: img = cv2.resize(img, (2000, round(img.shape[0] * 2000 / img.shape[1])), interpolation=cv2.INTER_AREA)
    out = HERE / "panels" / f"{name}.png"; cv2.imwrite(str(out), img[:, :, ::-1]); print(out, img.shape)

if __name__ == "__main__":
    name, page, box, cfgs = sys.argv[1], sys.argv[2], tuple(map(int, sys.argv[3].split(","))), sys.argv[4].split(",")
    sizes = {}
    import glob
    for c in cfgs:
        j = WORK / "res" / "cfg" / c / f"{page}.json"
        if j.exists(): sizes[c] = "%dKB" % (json.load(open(j))["bytes"] / 1000)
    panel(name, page, box, cfgs, sys.argv[5] if len(sys.argv) > 5 else None, sizes)
