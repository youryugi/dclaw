#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_ridge.log
        python3 -u run_twin.py "$@" --params 7d --k 16 --screen 512 --merge_sd 0.1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_ridge.log; }
for s in 0 1; do run ridge25_s$s robust2_laplace_is 1024 $s --ridge 2.5 --ridge_k 2; done
for s in 0 1; do run ridge25k3_s$s robust2_laplace_is 1024 $s --ridge 2.5 --ridge_k 3; done
echo "$(date +%T) ALL DONE" >> results/logs/queue_ridge.log
