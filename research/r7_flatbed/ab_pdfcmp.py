"""usage: ab_pdfcmp.py NAME -- PDF outputs of run_ab differ in metadata (dates / ids) even when the pages are identical.
Compare page by page on the raw bytes of every image XObject + the content stream; rewrite ab_NAME.json with
`differing_pdf_pages` / `identical_after_ignoring_metadata`, and record the (file index, page) pairs that really differ."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pikepdf

H = Path(__file__).resolve().parent
name = sys.argv[1]
base = H / "work" / f"ab_{name}"
summ = json.load(open(H / f"ab_{name}.json"))


def page_sig(pg):
    parts = [bytes(pg.Contents.read_bytes()) if "/Contents" in pg and not isinstance(pg.Contents, pikepdf.Array)
             else b"".join(bytes(c.read_bytes()) for c in pg.Contents)]
    for k in sorted(pg.Resources.get("/XObject", {}).keys()):
        x = pg.Resources.XObject[k]
        parts.append(bytes(x.read_raw_bytes()))
        for m in ("/SMask", "/Mask"):
            if m in x:
                parts.append(bytes(x[m].read_raw_bytes()))
    return parts


def cmp(i):
    o = next(iter((base / "old").glob(f"{i:04d}.*")), None)
    if o is None:
        return i, None, []
    n = base / "new" / o.name
    if o.suffix != ".pdf":
        return i, "image", []
    try:
        with pikepdf.open(o) as a, pikepdf.open(n) as b:
            if len(a.pages) != len(b.pages):
                return i, "pagecount", []
            bad = [k for k, (p, q) in enumerate(zip(a.pages, b.pages)) if page_sig(p) != page_sig(q)]
            return i, "pdf", bad
    except Exception as e:  # noqa
        return i, "err " + repr(e)[:60], []


def main():
    with ProcessPoolExecutor(8) as ex:
        rs = list(ex.map(cmp, summ["differing_index"]))
    real = [(i, k) for i, kind, bad in rs for k in bad]
    summ["pdf_differing_inputs_only_metadata"] = sum(1 for i, kind, bad in rs if kind == "pdf" and not bad)
    summ["inputs_with_real_page_differences"] = sorted({i for i, kind, bad in rs if bad or kind in ("image", "pagecount")})
    summ["real_page_differences"] = len(real)
    summ["real_pairs"] = real
    (H / f"ab_{name}.json").write_text(json.dumps(summ, indent=1))
    print({k: v for k, v in summ.items() if k not in ("differing_index", "real_pairs", "inputs_with_real_page_differences")},
          len(summ["inputs_with_real_page_differences"]))


if __name__ == "__main__":
    main()
