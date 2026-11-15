#!/usr/bin/env bash
cd "$(dirname "$0")/../experiments"
(for seed in 0 1 2; do ./sweep.sh recall.py 1 1 "--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed" ours_k4; done) &
./sweep.sh train_lm.py 1 1 "--steps 1500 --eval-every 250 --lr 2e-3" ours_k4 &
wait; echo QUEUE3_DONE
