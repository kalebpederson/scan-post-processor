"""Byte-identity check of two output directories, ignoring only the trailer /ID (random per run, also between two runs of
unmodified code).  Usage: cmp_ids.py DIR_A DIR_B"""
import hashlib
import re
import sys
from pathlib import Path

a, b = Path(sys.argv[1]), Path(sys.argv[2])
ok = True
for f in sorted(a.glob("*.pdf")):
    g = b / f.name
    if not g.exists():
        print("missing in B:", f.name)
        ok = False
        continue
    x, y = f.read_bytes(), g.read_bytes()
    strip = lambda d: re.sub(rb"/ID \[<[0-9a-f]+><[0-9a-f]+>\]", b"/ID []", d)
    same = strip(x) == strip(y)
    ok &= same
    print(f"{f.name} {len(x):>9} {len(y):>9} {'IDENTICAL (modulo /ID)' if same else 'DIFFERENT'}  sha256(stripped)={hashlib.sha256(strip(x)).hexdigest()[:16]}")
print("ALL IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
