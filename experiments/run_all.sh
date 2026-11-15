#!/usr/bin/env bash
# Reproduces every run reported in the README (sequentially; ~10 CPU-hours on 4 cores).
# Each call writes results/<task>/<run>.json; recall.py skips runs whose result exists.
set -e
cd "$(dirname "$0")"
R="--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3"
for seed in 0 1 2; do
  ./sweep.sh recall.py 4 1 "$R --seed $seed" \
    ours ours_k4 csa spk_softmax spikformer ssa_norm sdt wta wta3 ann \
    ours_nocomp ours_nonorm ours_binqk ours_nopbn spikformer_nopbn spikformer_cdglu
  ./sweep.sh recall.py 4 1 "--steps 4000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed $seed --tag long" \
    ours spk_softmax csa ours_nocomp spikformer wta
done
./sweep.sh train_lm.py 4 1 "--steps 1500 --eval-every 250 --lr 2e-3" ours_k4 ours ann spikformer wta spk_softmax sdt
./sweep.sh train_vision.py 4 1 "--dim 96 --depth 4 --heads 4 --epochs 5 --lr 2e-3" ours_k4 ours spikformer sdt ann
python3 summarize.py && python3 plots.py && python3 make_report.py ../docs/report.html
