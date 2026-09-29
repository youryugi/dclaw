#!/bin/sh
# Bimodal (single gauge, h@1) design: where does robust Laplace break? Logs in results/logs/
cd "$(dirname "$0")"
while pgrep -f bimodal_reference.py > /dev/null; do sleep 20; done
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_bimodal.log
        python3 -u run_twin.py "$@" --obs h@1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_bimodal.log; }
for k in 1 2 4; do for s in 0 1 2 3 4; do run bm_is_k${k}_s$s robust_laplace_is 1024 $s --k $k; done; done
for s in 0 1 2; do run bm_temper_k1_s$s robust_laplace 1024 $s --k 1; done
run bm_smcrw smc_rw 512 0
echo "$(date +%T) ALL DONE" >> results/logs/queue_bimodal.log
