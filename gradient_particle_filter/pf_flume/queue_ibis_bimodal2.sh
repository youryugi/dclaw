#!/bin/sh
cd "$(dirname "$0")"
run() { name=$1; shift; echo "$(date +%T) start $name" >> results/logs/queue_ibis_bimodal2.log
        python3 -u run_twin.py "$@" --obs h@1 > results/logs/$name.log 2>&1
        echo "$(date +%T) done  $name (exit $?)" >> results/logs/queue_ibis_bimodal2.log; }
run ib2_multi_s0 ibis_laplace_multi 512 0 --moves 2
for s in 1 2 3; do
  run ib2_multi_s$s   ibis_laplace_multi 512 $s --moves 2
  run ib2_laplace_s$s ibis_laplace 512 $s --moves 2
  run ib2_rw_s$s      ibis_rw 512 $s --moves 5
done
echo "$(date +%T) ALL DONE" >> results/logs/queue_ibis_bimodal2.log
