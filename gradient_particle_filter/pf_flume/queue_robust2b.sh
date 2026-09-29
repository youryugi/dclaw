#!/bin/sh
# improved robust Laplace (16 diverse starts, batched GN, converged modes only)
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_robust2b.log
        python3 -u run_twin.py "$@" > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_robust2b.log; }
for s in 0 1 2 3 4; do run r2b_h1_s$s robust2_laplace_is 1024 $s --obs h@1 --k 16 --screen 512; done
for s in 0 1; do run r2b_full_s$s robust2_laplace_is 1024 $s --k 16 --screen 512; done
echo "$(date +%T) ALL DONE" >> results/logs/queue_robust2b.log
