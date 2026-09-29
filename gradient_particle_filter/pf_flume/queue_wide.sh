#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_wide.log
        python3 -u run_twin.py "$@" --params 7d --k 16 --screen 512 --merge_sd 0.1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_wide.log; }
for s in 0 1 2; do run wide4096_s$s robust2_laplace_is 1024 $s --wide 4096; done
echo "$(date +%T) ALL DONE" >> results/logs/queue_wide.log
