"""Full-pipeline A/B on a directory of scans: previous code (git HEAD before the r9 change, extracted to work/oldsrc) vs the
working tree.  Both outputs are written to work/ab_NAME/{old,new}/ and compared byte for byte.

usage: run_ab.py SRC_DIR NAME [--jobs 8] [--only EXT,...]
Writes research/r9_holes/ab_NAME.json: counts of identical / differing outputs (no file names)."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PY = sys.executable
src, name = Path(sys.argv[1]), sys.argv[2]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".pdf"}
               and not p.name.startswith("contact_"))
base = HERE / "work" / f"ab_{name}"
for v in ("old", "new"):
    (base / v).mkdir(parents=True, exist_ok=True)


def run(args):
    i, p = args
    res = {}
    for v, pp in (("old", HERE / "work" / "oldsrc" / "src"), ("new", ROOT / "src")):
        out = base / v / f"{i:04d}{p.suffix.lower()}"
        env = dict(os.environ, PYTHONPATH=str(pp), SCANPP_NOTIFY="0")
        r = subprocess.run([PY, "-m", "scanpp", str(p), "-o", str(out), "--no-backup", "--foreground", "--jobs", "1"], env=env, capture_output=True, text=True)
        res[v] = (r.returncode, out)
    ok = all(rc == 0 and o.exists() for rc, o in res.values())
    same = ok and res["old"][1].read_bytes() == res["new"][1].read_bytes()
    if same:                                   # keep the disk clean: identical outputs are not needed any more
        for _, o in res.values():
            o.unlink()
    return i, ok, same


with ThreadPoolExecutor(jobs) as ex:
    rs = list(ex.map(run, enumerate(files)))
summ = dict(set=name, inputs=len(files), failed=sum(1 for _, ok, _ in rs if not ok),
            identical=sum(1 for _, ok, s in rs if ok and s), differing=sum(1 for _, ok, s in rs if ok and not s),
            differing_index=[i for i, ok, s in rs if ok and not s])
(HERE / f"ab_{name}.json").write_text(json.dumps(summ, indent=1))
(HERE / "work" / f"ab_{name}_files.json").write_text(json.dumps({i: str(p) for i, p in enumerate(files)}))
print(summ)
