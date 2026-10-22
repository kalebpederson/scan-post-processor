"""Wall time of processing the 11 sample PDFs (38 output pages) at each level: per pass the sum over files; report min and median over passes,
and the ratio to level 5.  Writes timing_summary.json."""
import json
import statistics
from pathlib import Path

d = Path(__file__).resolve().parent / "work" / "timing"
res = {}
for q in range(1, 11):
    passes = []
    for f in sorted(d.glob(f"pass*_q{q}.times")):
        rows = [l.split() for l in f.read_text().splitlines() if l.strip()]
        passes.append(sum(float(r[1]) for r in rows))
    res[q] = passes
print("| level | wall s per pass (11 PDFs) | min | median | min / level-5 min |")
print("|---|---|---|---|---|")
out = {}
for q, p in res.items():
    out[q] = {"passes": p, "min": min(p), "median": statistics.median(p), "rel_min": min(p) / min(res[5])}
    print(f"| {q} | {', '.join(f'{x:.1f}' for x in p)} | {min(p):.1f} | {statistics.median(p):.1f} | {min(p) / min(res[5]):.2f} |")
json.dump(out, open(Path(__file__).resolve().parent / "timing_summary.json", "w"), indent=1)
