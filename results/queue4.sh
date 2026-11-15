#!/usr/bin/env bash
cd "$(dirname "$0")/../experiments"
./sweep.sh train_vision.py 1 1 "--dim 96 --depth 4 --heads 4 --epochs 5 --lr 2e-3" ours_k4
echo QUEUE4_DONE
