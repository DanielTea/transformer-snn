# ShuntFormer v11 research handoff

The release restores eight endpoints and their two v10 parents.
Six endpoints are matched continuations on identical training
windows. Two more carry fast synapses fitted on top of a trained
decoder without changing its weights.
[Model card](../models/spiking-byte-lm-v11/README.md) ·
[Measurements, circuits and controls](SURPRISE_CLOCK.md)

## Restore the checkpoints

Use Python 3.12 and install `requirements.txt`. For dataset
preparation install `requirements-benchmarks.txt` as well.
The recorded runs used an Apple M3 Max with macOS 26.6.2,
Python 3.12.8, PyTorch 2.15.0 and NumPy 2.5.4. The figures
use Matplotlib 3.11.2.

Download and verify the archive from the repository root.

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v11
  for extension in sha256 tar.gz; do
    asset="spiking-byte-lm-v11-checkpoints.$extension"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
  done
  (cd checkpoints/downloads && shasum -a 256 -c spiking-byte-lm-v11-checkpoints.sha256)
  tar -xzf checkpoints/downloads/spiking-byte-lm-v11-checkpoints.tar.gz
  shasum -a 256 -c checkpoints/SHA256SUMS-v11
)
```

The endpoints are in `checkpoints/surprise-clock-study` and
the two parents are in `checkpoints/residual-scale-study`.
The manifest associates every checkpoint with its training
record. A decoder with fast synapses also names the trained
decoder under it and the record of its fit.
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v11) ·
[Exact checkpoint manifest](../models/spiking-byte-lm-v11/manifest.json)

The bundled model is the 25.79M decoder with fast synapses,
`surprise-clock-study/control-synapses_s10007.pt`. Its weights
are those of the `control` endpoint. The checkpoint adds the
synapse geometry and the 24 fitted strengths. The two decoders
with a surprise clock are in the archive as controls.

## Replay and rescore

Prepare enwik8, the PG-19 calibration and held-out books, the
fixed transfer suite, and the expanded training catalogue.
[Dataset commands](RESIDUAL_SCALE.md#reproduce-the-data-and-launches)

One script replays the study in order. It measures the frozen
v10 decoder, trains the six arms, fits and attaches the fast
synapses, scores every endpoint, and runs the fixed-weight
probes and the transfer suite. Every step keeps an existing
record, so the same command rescores restored checkpoints.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
study=checkpoints/surprise-clock-study
mkdir -p "$study/results"
cp results/event_lm/surprise_clock/*_s10007_surprise_clock_*.json "$study/results/"
bash experiments/run_surprise_clock.sh "$study"
```

Omit the `cp` line to train the six arms again from the v10
parents. Each run initializes a fresh AdamW optimizer and checks
its parent's hash and the mixture catalogue before training. A
single arm replays from its record.

```bash
record=results/event_lm/surprise_clock/event_correction_code_s10007_surprise_clock_clock.json
python experiments/audit_mixture_pg19_overlap.py "$record" \
  --json checkpoints/work/pg19-overlap-v11.json
python experiments/replay_event_run.py "$record" --enwik8 data/enwik8 \
  --training-mixture data/training-v1/prepared/catalogue.json \
  --device mps --output-dir checkpoints/work/replay-v11-clock
```

The warm audits check that each grown decoder reproduces its
parent's output counts before training.

```bash
parents=checkpoints/residual-scale-study
python experiments/audit_residual_growth.py --parent "$parents/wide-snn_s9631.pt" \
  --enwik8 data/enwik8 --device cpu --depth 15 --clock-start 11 --clock-threshold 35 \
  --json checkpoints/work/warm_clock.json
python experiments/audit_residual_growth.py --parent "$parents/wide-snn_s9631.pt" \
  --enwik8 data/enwik8 --device cpu --depth 15 --clock-start 11 --clock-threshold 5 \
  --clock-source periodic --json checkpoints/work/warm_stride.json
python experiments/audit_residual_growth.py --parent "$parents/wide-snn_s9631.pt" \
  --enwik8 data/enwik8 --device cpu --depth 15 --json checkpoints/work/warm_depth.json
python experiments/audit_transformer_growth.py --parent "$parents/wide-transformer_s9631.pt" \
  --enwik8 data/enwik8 --mlp-ratio 42.216796875 --seed 10007 --device cpu \
  --json checkpoints/work/warm_transformer.json
```

The fast synapses are fitted without gradient training of the
decoder. The fit records one pass of the frozen decoder over
enwik8 validation, replays the synapses on its error spikes and
solves for 24 integer strengths. `--score-test` replays the
test split with the final strengths.

```bash
python experiments/fit_fast_synapses.py --enwik8 data/enwik8 --device mps \
  --cache-dir "$study/cache" --checkpoint "$study/control_s10007.pt" \
  --groups 4 --cells 1024 --gate 1 --score-test \
  --json checkpoints/work/synapses_control.json
```

## Regenerate the reports

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v11 --release-dir checkpoints/release-v11
```

The [leaderboard](LEADERBOARD.md) shows the eight endpoints and
the fixed transfer suite.
