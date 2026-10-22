#!/bin/bash
# Wall-time repeat on a shared machine: 3 passes over all levels, results in work/timing/pass<N>_q<Q>.times (min over passes is reported).
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$here/work/timing"
for pass in 1 2 3; do
  for q in 5 1 2 3 4 6 7 8 9 10; do
    "$here/run_set.sh" "$here/work/timing/o" --quality $q > "$here/work/timing/pass${pass}_q$q.times"
  done
done
echo done
