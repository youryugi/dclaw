#!/bin/sh
# 7-parameter set, full observations
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_p7.log
        python3 -u run_twin.py "$@" --params 7 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_p7.log; }
for s in 0 1; do run p7_r2_s$s robust2_laplace_is 1024 $s --k 16 --screen 512; done
run p7_bootstrap bootstrap 4096 0
run p7_smcrw smc_rw 512 0
run p7_reference robust2_laplace_is 8192 9 --k 16 --screen 512
echo "$(date +%T) ALL DONE" >> results/logs/queue_p7.log
