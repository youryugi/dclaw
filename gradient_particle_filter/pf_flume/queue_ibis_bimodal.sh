#!/bin/sh
# IBIS on the bimodal single-gauge design
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_ibis_bimodal.log
        python3 -u run_twin.py "$@" --obs h@1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_ibis_bimodal.log; }
run ib_laplace       ibis_laplace 512 0 --moves 2
run ib_laplace_multi ibis_laplace_multi 512 0 --moves 2
run ib_rw            ibis_rw 512 0 --moves 5
echo "$(date +%T) ALL DONE" >> results/logs/queue_ibis_bimodal.log
