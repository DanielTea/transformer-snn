#!/usr/bin/env bash
cd "$(dirname "$0")"
for seed in 0 1 2; do
  ./sweep.sh recall.py 3 1 "--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed" \
    ours csa spk_softmax spikformer ssa_norm sdt ann ours_nocomp ours_nonorm ours_binqk ours_nopbn spikformer_nopbn spikformer_cdglu
done
