#!/usr/bin/env bash
# Longer-budget recall (4000 steps): does each variant eventually solve the task?
cd "$(dirname "$0")/../experiments"
while pgrep -f run_fixups.sh > /dev/null; do sleep 30; done
for seed in 0 1 2; do
  ./sweep.sh recall.py 1 1 "--steps 4000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed --tag long" \
    ours spk_softmax csa ours_nocomp spikformer wta
done
echo LONG_DONE
