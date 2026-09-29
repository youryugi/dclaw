#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_pad.log
        python3 -u run_twin.py "$@" > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_pad.log; }
for s in 0 1 2; do run pad_r2_h1_s$s robust2_laplace_is 1024 $s --obs h@1 --k 16 --screen 512; done
for s in 0 1; do run pad_r2_full_s$s robust2_laplace_is 1024 $s --k 16 --screen 512; done
for s in 0 1 2 3; do run pad_ibmulti_s$s ibis_laplace_multi 512 $s --obs h@1 --moves 2; done
echo "$(date +%T) ALL DONE" >> results/logs/queue_pad.log
