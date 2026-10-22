"""usage: ab_check.py NAME -- cross-check run_ab (pipeline outputs) against eval_edges (edge-level changes)."""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
n = sys.argv[1]
a = json.load(open(H / "work" / f"ab_{n}_files.json"))
ab = json.load(open(H / f"ab_{n}.json"))
b = ab.get("inputs_with_real_page_differences", ab["differing_index"])
e = json.load(open(H / f"eval_{n}.json"))
m = json.load(open(H / "work" / f"files_{n}.json"))
ech = {m[str(r["fid"])] for r in e["changed"]}
dif = {a[str(i)] for i in b}
print("pipeline-differing:", len(dif), "edge-changed:", len(ech), "| differing but not edge-changed:", sorted(x.split('/')[-1] for x in dif - ech))
