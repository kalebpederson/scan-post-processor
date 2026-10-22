"""List plain-scan PDFs in the (private) samples dir: size, pages, image px of page 1 (to see the native ppi)."""
import glob
import os
import sys

import pikepdf

from scanpp import pdfio

root = sys.argv[1]
rows = []
for f in glob.glob(os.path.join(root, "**/*.pdf"), recursive=True):
    try:
        with pikepdf.open(f) as p:
            ok, why = pdfio.classify_pdf(p)
            if not ok:
                continue
            img = next(iter(p.pages[0].resources.XObject.values()))
            mb = [float(x) for x in p.pages[0].MediaBox]
            ppi = int(img.Width) / ((mb[2] - mb[0]) / 72)
            rows.append((os.path.getsize(f), len(p.pages), round(ppi), f))
    except Exception:
        pass
rows.sort()
for r in rows:
    print(*r, sep="\t")
