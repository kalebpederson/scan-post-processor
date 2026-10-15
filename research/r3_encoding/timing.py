"""Single-thread, sequential encode timing (median of 3 pages x 2 reps; machine is shared so treat as upper bound)."""
import time, statistics as st
from config import *
from configs import C
from encoders import build_pdf
import e0
PG = ["single13", "pair09_front", "single02"]
out = {}
imgs = {p: load_clean(p) for p in PG}
for c in ["E1_j150_q40_420", "E1_j200_q55_420", "E1_j200_q85_444", "E2_p150_k8_km", "E2_p200_k16_km", "E3_c75_q40", "E3_base", "E3_c75_q40_hl", "E4_s2", "E4_s7"]:
    ts = []
    for p in PG:
        for _ in range(2):
            t = time.perf_counter(); pg = C[c]["fn"](imgs[p], **C[c]["kw"]); build_pdf([pg], str(WORK / "tmp" / "t.pdf")); ts.append(time.perf_counter() - t)
    out[c] = st.median(ts)
P = {p["name"]: p for p in pages()}
ts = []
os.environ["MAGICK_THREAD_LIMIT"] = "1"
for p in PG:
    t = time.perf_counter(); e0.run_e0(P[p], str(WORK / "tmp")); ts.append(time.perf_counter() - t)
out["E0(literal cmd)"] = st.median(ts)
json.dump(out, open(WORK / "res" / "timing.json", "w"), indent=1); print(out)
