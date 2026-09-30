#!/bin/sh
cd "$(dirname "$0")"
t() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_esmda.log
      python3 -u run_twin.py "$@" > results/logs/$name.log 2>&1
      echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_esmda.log; }
r() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_esmda.log
      python3 -u run_real.py "$@" > results/logs/$name.log 2>&1
      echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_esmda.log; }
t es_full_k4_s0 esmda 512 0 --esmda_k 4
t es_full_k4_s1 esmda 512 1 --esmda_k 4
t es_full_k8_s0 esmda 512 0 --esmda_k 8
t es_bimodal_k4 esmda 512 0 --esmda_k 4 --obs h@1
t es_bimodal_k8 esmda 512 0 --esmda_k 8 --obs h@1
t es_p7d_k4     esmda 512 0 --esmda_k 4 --params 7d
t es_p7d_k8     esmda 512 0 --esmda_k 8 --params 7d
r es_real_k4    esmda 512 0 --esmda_k 4
r es_real_k8    esmda 512 0 --esmda_k 8
echo "$(date +%T) ALL DONE" >> results/logs/queue_esmda.log
