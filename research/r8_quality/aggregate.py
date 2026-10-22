"""Aggregate a study jsonl: mean per candidate over pages (and, with --by-scheme, per page class of the level-5 rule).

    python aggregate.py work/sweep.jsonl [--json out.json] [--by-scheme]
"""
import argparse
import collections
import json

import numpy as np

KEYS = ["ssim", "psnr", "de_ink", "de_col", "edge_sharp", "loss_solid", "loss_faint", "speckle", "ocr_recall", "enc_s"]


def load(path):
    return [json.loads(line) for line in open(path)]


def agg(rows):
    out = {}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cand"]].append(r)
    for c, rs in by.items():
        d = {"pages": len(rs), "kb": float(np.mean([r["bytes"] for r in rs])) / 1000}
        d["kb_median"] = float(np.median([r["bytes"] for r in rs])) / 1000
        d["kb_p90"] = float(np.percentile([r["bytes"] for r in rs], 90)) / 1000
        for k in KEYS:
            v = [r[k] for r in rs if r.get(k) is not None]
            d[k] = float(np.mean(v)) if v else None
        d["filters"] = sorted({f for r in rs for f in r["filters"]})
        out[c] = d
    return out


def fmt(x, n=3):
    return "-" if x is None else f"{x:.{n}f}"


def table(a):
    hdr = "| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |"
    lines = [hdr, "|" + "---|" * 15]
    for c, d in sorted(a.items(), key=lambda kv: kv[1]["kb"]):
        lines.append(f"| {c} | {d['pages']} | {d['kb']:.1f} | {d['kb_median']:.1f} | {d['kb_p90']:.1f} | {fmt(d['ssim'], 4)} | {fmt(d['psnr'], 2)} | "
                     f"{fmt(d['de_ink'], 2)} | {fmt(d['de_col'], 2)} | {fmt(d['edge_sharp'], 3)} | {fmt(d['loss_solid'], 4)} | "
                     f"{fmt(d['loss_faint'], 3)} | {fmt(d['speckle'], 4)} | {fmt(d['ocr_recall'], 3)} | {fmt(d['enc_s'], 2)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--json")
    ap.add_argument("--by-scheme", action="store_true")
    a = ap.parse_args()
    rows = load(a.jsonl)
    res = {"all": agg(rows)}
    print(f"all pages ({len({r['page'] for r in rows})}):")
    print(table(res["all"]))
    if a.by_scheme:
        for s in sorted({r["scheme5"] for r in rows}):
            sub = [r for r in rows if r["scheme5"] == s]
            res[s] = agg(sub)
            print(f"\nclass {s} ({len({r['page'] for r in sub})} pages):")
            print(table(res[s]))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
