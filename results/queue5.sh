#!/usr/bin/env bash
cd "$(dirname "$0")/../experiments"
for seed in 2 1 0; do ./sweep.sh recall.py 1 1 "--steps 4000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed --tag long" wta spikformer ours_nocomp csa spk_softmax ours; done
echo QUEUE5_DONE
