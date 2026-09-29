#!/bin/sh
cd "$(dirname "$0")"
R=results/full_noise0_p7
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_polish.log
        python3 -u mcmc_polish.py "$@" > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_polish.log; }
run polish_smcrw_rw50       $R/smc_rw_N512_s0.npz rw 50
run polish_r2smc_rw50       $R/robust2_laplace_smc_k16_m512_merge0.1_N1024_s0.npz rw 50
run esjd_rw20_256           $R/smc_rw_N512_s0.npz rw 20 256
run esjd_mala5_256          $R/smc_rw_N512_s0.npz mala 5 256
echo "$(date +%T) ALL DONE" >> results/logs/queue_polish.log
