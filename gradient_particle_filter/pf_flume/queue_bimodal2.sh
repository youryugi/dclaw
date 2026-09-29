#!/bin/sh
# rerun k = 2, 4 after fixing the mode-merge criterion
cd "$(dirname "$0")"
until grep -q "ALL DONE" results/logs/queue_bimodal.log 2>/dev/null; do sleep 30; done
mkdir -p results/h_1_noise0/before_merge_fix && mv results/h_1_noise0/robust_laplace_is_k2_* results/h_1_noise0/robust_laplace_is_k4_* results/h_1_noise0/before_merge_fix/
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_bimodal2.log
        python3 -u run_twin.py "$@" --obs h@1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_bimodal2.log; }
for k in 2 4; do for s in 0 1 2 3 4; do run bm2_is_k${k}_s$s robust_laplace_is 1024 $s --k $k; done; done
run bm2_temper_k4_s0 robust_laplace 1024 0 --k 4
echo "$(date +%T) ALL DONE" >> results/logs/queue_bimodal2.log
