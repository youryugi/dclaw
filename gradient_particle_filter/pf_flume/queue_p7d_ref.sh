#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_p7d_ref.log
        python3 -u run_twin.py "$@" --params 7d > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_p7d_ref.log; }
run p7d_ref_is   robust2_laplace_is 16384 5 --k 16 --screen 512 --merge_sd 0.1
run p7d_ref_smc  smc_rw 1024 5 --moves 10
echo "$(date +%T) ALL DONE" >> results/logs/queue_p7d_ref.log
