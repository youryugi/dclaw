#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_real2.log
        python3 -u run_real.py "$@" > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_real2.log; }
run real_r2_all     robust2 1024 0 --calib 012
run real_r2_frac03  robust2 1024 0 --calib 01 --frac 0.3
echo "$(date +%T) ALL DONE" >> results/logs/queue_real2.log
