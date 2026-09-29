#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_real.log
        python3 -u run_real.py "$@" > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_real.log; }
run real_r2_s0   robust2 1024 0
run real_r2_s1   robust2 1024 1
run real_smcrw   smc_rw 512 0
echo "$(date +%T) ALL DONE" >> results/logs/queue_real.log
