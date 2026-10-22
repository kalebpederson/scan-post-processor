"""Summarise run_levels.sh output: per level, total output bytes of the sample PDFs (and relative to level 5), pages, KB/page,
encode-stage CPU and wall time.  Prints a markdown table and writes levels_summary.json (aggregate numbers only)."""
import json
import sys
from pathlib import Path

import pikepdf

work = Path(__file__).resolve().parent / "work"
levels = {}
for d in sorted(work.glob("lv*")):
    if not d.is_dir():
        continue
    tag = d.name[2:]
    pdfs = sorted(d.glob("s*.pdf"))
    if not pdfs:
        continue
    tot = sum(p.stat().st_size for p in pdfs)
    pages = 0
    enc_cpu = clean_cpu = 0.0
    per = {}
    for p in pdfs:
        rep = json.loads((d / (p.stem + ".json")).read_text())
        for f in rep["files"]:
            pages += f["pages_out"]
            enc_cpu += f["stage_cpu"].get("encode", 0.0)
            clean_cpu += sum(f["stage_cpu"].values())
        per[p.stem] = p.stat().st_size
    wall = 0.0
    tf = work / f"lv{tag}.times"
    for line in tf.read_text().split():
        pass
    wall = sum(float(l.split()[1]) for l in tf.read_text().splitlines() if l.strip())
    levels[tag] = {"bytes": tot, "pages": pages, "kb_page": tot / pages / 1000, "encode_cpu_s": enc_cpu, "total_cpu_s": clean_cpu,
                   "wall_s": wall, "per_pdf": per}
base = levels["5"]
print("| level | total KB | KB/page | x level 5 | min..max per-PDF x L5 | encode CPU s | total CPU s | wall s |")
print("|---|---|---|---|---|---|---|---|")
out = {}
for tag in sorted(levels, key=lambda t: (int(t.rstrip("b")), t)):
    v = levels[tag]
    rel = [v["per_pdf"][k] / base["per_pdf"][k] for k in base["per_pdf"]]
    print(f"| {tag} | {v['bytes'] / 1000:.0f} | {v['kb_page']:.1f} | {v['bytes'] / base['bytes']:.2f} | {min(rel):.2f}..{max(rel):.2f} | "
          f"{v['encode_cpu_s']:.1f} | {v['total_cpu_s']:.1f} | {v['wall_s']:.1f} |")
    out[tag] = {k: v[k] for k in ("bytes", "pages", "kb_page", "encode_cpu_s", "total_cpu_s", "wall_s")}
    out[tag]["rel_to_L5"] = v["bytes"] / base["bytes"]
    out[tag]["rel_min"], out[tag]["rel_max"] = min(rel), max(rel)
json.dump(out, open(Path(__file__).resolve().parent / "levels_summary.json", "w"), indent=1)
