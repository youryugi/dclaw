#!/bin/sh
# robustness (item 1) and sequential assimilation (item 2); logs in results/logs/
cd "$(dirname "$0")"
mkdir -p results/logs
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue.log
        python3 -u run_twin.py "$@" > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue.log; }
run A_robust_full0            robust_laplace 1024 0
for s in 1 2 3 4; do run A_robustis_full0_s$s robust_laplace_is 1024 $s; done
run C_bootstrap_h0            bootstrap 4096 0 --obs h
run C_robust_h0               robust_laplace 1024 0 --obs h
run C_smcrw_h0                smc_rw 512 0 --obs h
run D_ibis_laplace_full0      ibis_laplace 512 0 --moves 2
run D_ibis_rw_full0           ibis_rw 512 0 --moves 5
for k in 1 2; do run B_robust_full$k robust_laplace 1024 0 --noise $k; run B_smcrw_full$k smc_rw 512 0 --noise $k; done
echo "$(date +%T) ALL DONE" >> results/logs/queue.log
