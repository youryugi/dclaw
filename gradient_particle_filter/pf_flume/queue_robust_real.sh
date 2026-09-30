#!/bin/sh
cd "$(dirname "$0")"
r() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_robust_real.log
      python3 -u run_real.py "$@" > results/logs/$name.log 2>&1
      echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_robust_real.log; }
t() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_robust_real.log
      python3 -u run_twin.py "$@" > results/logs/$name.log 2>&1
      echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_robust_real.log; }
r rr_pbnone     robust2 1024 0 --pbconv none
r rr_pbcos      robust2 1024 0 --pbconv cos
r rr_wide       robust2 1024 0 --params 3w
r rr_ar1        robust2 1024 0 --ar1 0.9
r rr_wide_ar1   robust2 1024 0 --params 3w --ar1 0.9
r rr_wide_ar1_fine robust2 1024 0 --params 3w --ar1 0.9 --dx 0.0625
r rr_wide_ar1_all  robust2 1024 0 --params 3w --ar1 0.9 --calib 012
t es2_full_n2048k8   esmda 2048 0 --esmda_k 8
t es2_full_n512k16   esmda 512 0 --esmda_k 16
t es2_bim_n2048k8    esmda 2048 0 --esmda_k 8 --obs h@1
t es2_bim_n512k16    esmda 512 0 --esmda_k 16 --obs h@1
t es2_p7d_n2048k8    esmda 2048 0 --esmda_k 8 --params 7d
t es2_p7d_n512k16    esmda 512 0 --esmda_k 16 --params 7d
echo "$(date +%T) ALL DONE" >> results/logs/queue_robust_real.log
