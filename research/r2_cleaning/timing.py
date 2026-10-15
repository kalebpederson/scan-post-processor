"""single-threaded timing (run alone on an otherwise idle core): ms/MP + seconds for a 300 dpi letter page."""
import sys, time, json; sys.path.insert(0,'.')
from common import *
from clean import clean_page
def bench(fn, rgb, n=5):
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(rgb); ts.append(time.perf_counter() - t)
    return float(np.median(ts)), float(np.min(ts))
if __name__ == "__main__":
    params = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = {}
    for nm, dpi in (("single13.jpg", 300), ("single05.jpg", 200), ("single09.jpg", 150)):
        rgb = load_rgb(CORPUS / nm); mp = rgb.shape[0] * rgb.shape[1] / 1e6
        med, mn = bench(lambda r: clean_page(r, dpi, **params), rgb)
        out[f"{dpi}dpi_{rgb.shape[1]}x{rgb.shape[0]}"] = dict(sec_median=round(med, 3), sec_min=round(mn, 3), ms_per_mp=round(med * 1000 / mp, 1))
        medb, _ = bench(b0, rgb); out[f"{dpi}dpi_B0_numpy"] = dict(sec_median=round(medb, 3), ms_per_mp=round(medb * 1000 / mp, 1))
    print(json.dumps(out, indent=1))
