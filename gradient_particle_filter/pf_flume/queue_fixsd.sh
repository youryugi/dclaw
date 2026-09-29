#!/bin/sh
# fixing the under-dispersed 7-parameter (7d) Laplace IS
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_fixsd.log
        python3 -u run_twin.py "$@" --params 7d --k 16 --screen 512 --merge_sd 0.1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_fixsd.log; }
for s in 0 1; do run fix_wide_s$s  robust2_laplace_is  1024 $s --inflate 2.5 --dof 3; done
for s in 0 1; do run fix_adapt_s$s robust2_laplace_is  1024 $s --adapt 2; done
for s in 0 1; do run fix_temper_s$s robust2_laplace_smc 1024 $s; done
echo "$(date +%T) ALL DONE" >> results/logs/queue_fixsd.log
