"""R2 metric suite (research/r2_cleaning/evaluate.py) pointed at scanpp.clean.clean_page.
Usage: .venv/bin/python bench/eval_clean.py OUT.json [--quick]
Does not write into research/ (no bytecode, no pickles)."""
import os, sys, json
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R2 = os.path.join(ROOT, "research", "r2_cleaning")
os.chdir(R2); sys.path.insert(0, R2)
import cv2; cv2.setNumThreads(1)
import evaluate, run_eval
from scanpp import clean as pkg_clean
evaluate.clean_page = pkg_clean.clean_page          # run_method() looks the name up in evaluate's globals

CFGS = [("FINAL_flatten", "clean", {}), ("FINAL_keep", "clean", {}, "keep")]
if os.environ.get("EVAL_PARAMS"):      # extra clean_page kwargs for experiments, e.g. EVAL_PARAMS='{"pk_seed":0.25}' (replaces the defaults config)
    _p = json.loads(os.environ["EVAL_PARAMS"]); CFGS = [("FINAL_flatten", "clean", _p), ("FINAL_keep", "clean", _p, "keep")]
KEYS = ["pos_ret", "neg_surv_regions", "bleed_surv_pooled", "bleed_surv_macro", "bleed_dark_vis", "white_whitepaper",
        "white_whitepaper_min", "bk_clean", "keep_dE_mean", "ink_area_ratio", "ink_med", "ink_p90", "ink_mean",
        "hue_shift_mean", "hue_shift_p90", "chroma_ratio_med", "chroma_ratio_p10", "color_dL_med", "pos_contrast_ratio"]

if __name__ == "__main__":
    out = sys.argv[1]
    agg, res = run_eval.evaluate(CFGS, quick="--quick" in sys.argv, procs=int(os.environ.get("PROCS", 16)))
    summ = {c: {k: agg[c].get(k) for k in KEYS} for c in agg}
    for c in summ:
        summ[c]["pos_each"] = agg[c].get("pos_each")
    json.dump(summ, open(out, "w"), indent=1, default=float)
    for c in summ:
        print(c, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ[c].items() if k != "pos_each"})
