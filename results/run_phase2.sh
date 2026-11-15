#!/usr/bin/env bash
# Phase 2: starts when the recall sweep is done. Keeps 4 CPU jobs busy.
cd "$(dirname "$0")/../experiments"
while pgrep -f run_recall.sh > /dev/null; do sleep 15; done
# WTA baselines on recall (3 seeds), 1 core
( for seed in 0 1 2; do ./sweep.sh recall.py 1 1 "--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed" wta wta3; done ) &
# remaining LM variants, 1 core
./sweep.sh train_lm.py 1 1 "--steps 1500 --eval-every 250 --lr 2e-3" ann wta sdt spk_softmax &
# CIFAR-10, 2 cores
./sweep.sh train_vision.py 2 1 "--dim 96 --depth 4 --heads 4 --epochs 5 --lr 2e-3" ours spikformer sdt ann &
wait
