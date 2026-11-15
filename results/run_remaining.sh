#!/usr/bin/env bash
# Remaining sweeps, run detached (setsid nohup) so no session time limit kills them.
cd "$(dirname "$0")/../experiments"
R="--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3"
(
  ./sweep.sh recall.py 1 1 "$R --seed 2" ours_binqk ours_nopbn spikformer_nopbn spikformer_cdglu
  for seed in 0 1 2; do ./sweep.sh recall.py 1 1 "$R --seed $seed" wta wta3; done
) &
./sweep.sh train_lm.py 2 1 "--steps 1500 --eval-every 250 --lr 2e-3" ours ann spikformer wta spk_softmax sdt &
./sweep.sh train_vision.py 2 1 "--dim 96 --depth 4 --heads 4 --epochs 5 --lr 2e-3" ours spikformer sdt ann &
wait
echo ALL_DONE
