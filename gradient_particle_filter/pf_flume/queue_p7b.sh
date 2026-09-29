#!/bin/sh
cd "$(dirname "$0")"
until grep -q "ALL DONE" results/logs/queue_p7.log 2>/dev/null; do sleep 30; done
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_p7b.log
        python3 -u run_twin.py "$@" --params 7 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_p7b.log; }
run p7_r2_merge01_is  robust2_laplace_is  1024 0 --k 16 --screen 512 --merge_sd 0.1
run p7_r2_merge01_smc robust2_laplace_smc 1024 0 --k 16 --screen 512 --merge_sd 0.1
echo "$(date +%T) ALL DONE" >> results/logs/queue_p7b.log
