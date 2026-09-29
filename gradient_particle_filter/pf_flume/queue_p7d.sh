#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_p7d.log
        python3 -u run_twin.py "$@" --params 7d > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_p7d.log; }
run p7d_r2_is   robust2_laplace_is  1024 0 --k 16 --screen 512 --merge_sd 0.1
run p7d_r2_is1  robust2_laplace_is  1024 1 --k 16 --screen 512 --merge_sd 0.1
run p7d_smcrw   smc_rw 512 0
echo "$(date +%T) ALL DONE" >> results/logs/queue_p7d.log
