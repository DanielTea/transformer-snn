#!/usr/bin/env bash
cd "$(dirname "$0")/../experiments"
R="--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3"
rm -f ../results/recall/wta3_ilif4_T1_s0_P8.json
./sweep.sh recall.py 1 1 "$R --seed 0" wta wta3
for seed in 0 1 2; do ./sweep.sh recall.py 1 1 "--steps 4000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed --tag long" ours spk_softmax csa ours_nocomp spikformer wta; done
echo QUEUE2_DONE
