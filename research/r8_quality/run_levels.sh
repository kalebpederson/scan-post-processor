#!/bin/bash
# Runs run_set.sh for --quality 5 1 2 3 4 6 7 8 9 10 5 (level 5 twice, as a noise estimate); wall seconds per file in work/lv*.times
here=$(cd "$(dirname "$0")" && pwd)
for q in 5 1 2 3 4 6 7 8 9 10 5; do
  tag=lv$q
  [ -d "$here/work/$tag" ] && tag=lv${q}b
  "$here/run_set.sh" "$here/work/$tag" --quality $q > "$here/work/$tag.times"
done
echo done
