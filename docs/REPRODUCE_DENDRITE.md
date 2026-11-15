# Reproduce Dendrite

Run these commands from the repository root. Python 3.12 and PyTorch
2.15 were used for the recorded CPU experiments. The repository
contains the model code, experiment runners, result JSON and chart
source. Training writes checkpoints to the local `checkpoints/` directory.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

## Regenerate the reports

The result files are enough to render the figures and the browser
report. No datasets or checkpoints are needed.

```bash
python experiments/plot_dendrite.py
python experiments/make_report.py docs/report.html
```

Open [report.html](report.html) in a browser, or read the
[GitHub benchmark report](BENCHMARKS.md). The
[results index](../results/README.md) maps each panel to its source JSON.

## Recall with variable gaps

The command groups below use Bash arrays.

```bash
recall_args=(--keys 32 --pairs 4 --queries 4 --train-max-gap 4
  --eval-gaps 0 4 16 64 128 --eval-context-length 524 --eval-bs 16
  --eval-samples 512 --val-samples 256 --threads 1 --lr 0.003
  --steps 1200 --eval-every 300 --tag reproduce)

for seed in 0 1 2; do
  python experiments/recall_distance.py "${recall_args[@]}" \
    --variant dendrite_event --address-bits 6 --seed "$seed"
  python experiments/recall_distance.py "${recall_args[@]}" \
    --variant transformer --seed "$seed"
done
python experiments/recall_distance.py "${recall_args[@]}" \
  --variant dendrite --address-bits 6 --seed 0

python experiments/recall_distance.py --variant dendrite_event --address-bits 6 \
  --keys 32 --pairs 4 --queries 4 --train-max-gap 4 --eval-gaps 0 4 128 512 2048 \
  --eval-context-length 8204 --eval-bs 8 --eval-samples 128 --val-samples 256 \
  --anchor-eval-values --threads 1 --lr 0.003 --steps 1200 --eval-every 300 \
  --seed 0 --tag reproduce_long --checkpoint checkpoints/recall-event-reproduce.pt
```

The second run layout keeps value and query positions fixed across all
gaps. The runner uses different random seeds for training, validation
and test data. Keep the evaluation batch size fixed when comparing
sampled results.

## Language models and transfer

Download and extract the raw
[enwik8 corpus](https://www.mattmahoney.net/dc/textdata.html). The runners
verify the expected 100,000,000-byte file and SHA-1 before loading it.
Set `ENWIK8_PATH` to its location.

```bash
export ENWIK8_PATH=/absolute/path/to/enwik8
lm_args=(--dim 96 --depth 3 --heads 4 --ctx 1024 --bs 4 --steps 1000
  --warmup 100 --eval-every 250 --eval-bs 4 --eval-batches 10
  --final-eval-batches 64 --lr 0.003 --threads 1 --seed 0 --tag reproduce)

python experiments/train_lm.py "${lm_args[@]}" --variant transformer_local \
  --checkpoint checkpoints/local-reproduce.pt
python experiments/train_lm.py "${lm_args[@]}" --variant transformer_successor \
  --checkpoint checkpoints/successor-reproduce.pt

for seed in 0 1 2; do
  python experiments/train_successor.py --base checkpoints/local-reproduce.pt \
    --scaled-votes --spike-warmup 80 --steps 320 --ctx 2048 --bs 4 \
    --eval-context 4096 --eval-windows 128 --seed "$seed" --tag reproduce \
    --checkpoint "checkpoints/successor-calibrated-s${seed}.pt"
done

python experiments/prepare_pg19.py --output data/pg19-test8-128k
python experiments/eval_successor.py \
  --checkpoint checkpoints/successor-calibrated-s0.pt --chunk 4096 \
  --raw data/pg19-test8-128k/*.txt --json results/lm_successor/calibrated_reproduce.json
python experiments/eval_successor.py \
  --checkpoint checkpoints/successor-reproduce.pt --chunk 4096 \
  --raw data/pg19-test8-128k/*.txt --json results/lm_successor/end_to_end_reproduce.json
```

The calibrated run reports its frozen local backbone alongside the
successor readout. Use that local score as the independently trained
control for the from-scratch run. The `base_bpb` field from the
from-scratch checkpoint instead scores its jointly trained backbone
with the cache muted. To reproduce calibration with uniform spike
increments, omit `--scaled-votes` and `--spike-warmup 80` and use a fresh
result tag and checkpoint path.

## Source map

| Source | Responsibility |
|---|---|
| [dendritic.py](../snnformer/dendritic.py) | Spike addresses, fixed delay and bounded bucket readout |
| [dendritic_event.py](../snnformer/dendritic_event.py) | Learned capture and commit with a parallel training scan |
| [transformer_reference.py](../snnformer/transformer_reference.py) | Full and sliding-window Transformer controls, local streaming |
| [successor_memory.py](../snnformer/successor_memory.py) | Verified byte successor tables and streaming lookup |
| [successor_lm.py](../snnformer/successor_lm.py) | Spike readout, probability mixture and token-by-token evaluation |
| [recall_distance.py](../experiments/recall_distance.py) | Typed recall data, training and held-out distance evaluation |
| [train_lm.py](../experiments/train_lm.py) | Matched next-byte training and checkpoint output |
| [train_successor.py](../experiments/train_successor.py) | Frozen-backbone calibration and readout controls |
| [eval_successor.py](../experiments/eval_successor.py) | Whole-stream enwik8 and independent raw byte evaluation |
| [prepare_pg19.py](../experiments/prepare_pg19.py) | Checksum-verified PG-19 excerpt preparation |
| [plot_dendrite.py](../experiments/plot_dendrite.py) | Figures from recorded result JSON |
