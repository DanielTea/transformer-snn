#!/usr/bin/env bash
# Re-run the WTA recall runs that used the NaN-gradient version, once the main recall queue is idle.
cd "$(dirname "$0")/../experiments"
while pgrep -f "recall.py --variant wta" > /dev/null || pgrep -f "sweep.sh recall.py" > /dev/null; do sleep 20; done
rm -f ../results/recall/wta3_ilif4_T1_s0_P8.json ../results/recall/wta_ilif4_T1_s0_P8.json
./sweep.sh recall.py 1 1 "--steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed 0" wta wta3
echo FIXUPS_DONE
