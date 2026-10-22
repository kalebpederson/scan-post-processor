#!/bin/bash
# usage: run_set.sh OUTDIR [scanpp options...]
# Processes work/samples/s*.pdf with the scanpp of this checkout into OUTDIR; prints "name wall_s user_s sys_s" per file.
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
out=$1; shift
mkdir -p "$out"
export PYTHONPATH=$root/src
export SCANPP_LOG_DIR=$here/work/log
for f in "$here"/work/samples/s*.pdf; do
  b=$(basename "$f")
  /usr/bin/time -f "$b %e %U %S" -o "$out/$b.time" "${PYTHON:-python3}" -m scanpp --foreground --no-backup --no-notify --jobs 16 --report "$out/${b%.pdf}.json" "$@" "$f" -o "$out/$b" >/dev/null 2>&1 || echo "FAIL $b"
  cat "$out/$b.time"; rm -f "$out/$b.time"
done
