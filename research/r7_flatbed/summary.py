"""Print the headline numbers of eval_*.json / ab_*.json (used for REPORT.md)."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
for n in ("corpus", "samples"):
    e = json.load(open(H / f"eval_{n}.json"))
    a = json.load(open(H / f"ab_{n}.json"))
    ch = e["changed"]
    print(n, {k: v for k, v in e.items() if k != "changed"})
    print("  ab", {k: v for k, v in a.items() if k not in ("differing_index", "real_pairs", "inputs_with_real_page_differences")})
    print("  rules line/fringe/corner/hairline:", sum(1 for r in ch if r["line"]), sum(1 for r in ch if r["fringe"]),
          sum(1 for r in ch if r["corner"]), sum(1 for r in ch if r["hairline"]),
          "| only-hairline:", sum(1 for r in ch if r["hairline"] and not (r["line"] or r["fringe"] or r["corner"])),
          "| only-corner:", sum(1 for r in ch if r["corner"] and not (r["line"] or r["fringe"] or r["hairline"])))
