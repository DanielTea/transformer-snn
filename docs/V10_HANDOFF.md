# ShuntFormer v10 research handoff

The release restores five endpoints and their two v9 parents.
The bundled model uses the 1,408-wide spiking residual circuit
and the expanded training mixture. The matched study also
includes the original widths and the wide enwik8-only SNN.
[Model card](../models/spiking-byte-lm-v10/README.md) ·
[Experiment and corpus preparation](RESIDUAL_SCALE.md)

## Restore the checkpoints

Use Python 3.12 and install `requirements.txt`. For dataset
preparation install `requirements-benchmarks.txt` as well.
The recorded runs used an Apple M3 Max with macOS 26.6.2,
Python 3.12.8, PyTorch 2.15.0 and NumPy 2.5.4. Dataset
preparation used PyArrow 21.0.0, PySBD 0.3.4 and
pyahocorasick 2.1.0. The figures use Matplotlib 3.11.2.

Download and verify the archive from the repository root.

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v10
  for extension in sha256 tar.gz; do
    asset="spiking-byte-lm-v10-checkpoints.$extension"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
  done
  (cd checkpoints/downloads && shasum -a 256 -c spiking-byte-lm-v10-checkpoints.sha256)
  tar -xzf checkpoints/downloads/spiking-byte-lm-v10-checkpoints.tar.gz
  shasum -a 256 -c checkpoints/SHA256SUMS-v10
)
```

The endpoints are in `checkpoints/residual-scale-study` and
the two parents are in `checkpoints/trace-arrival-study`.
The source tree includes the 25.79M SNN weights and the manifest
that associates every checkpoint with its training record.
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v10) ·
[Exact checkpoint manifest](../models/spiking-byte-lm-v10/manifest.json)

## Replay and rescore

Prepare enwik8, the PG-19 calibration and held-out books, the
fixed transfer suite, and the expanded training catalogue.
The prepared catalogue and its window receipt have committed hashes.
[Dataset commands](RESIDUAL_SCALE.md#reproduce-the-data-and-launches)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
record=results/event_lm/residual_scale/event_correction_code_s9631_residual_scale_wide_snn.json
python experiments/audit_mixture_pg19_overlap.py "$record" \
  --json checkpoints/work/pg19-overlap-v10.json
python experiments/replay_event_run.py "$record" --enwik8 data/enwik8 \
  --training-mixture data/training-v1/prepared/catalogue.json \
  --device mps --output-dir checkpoints/work/replay-v10-wide
```

The replay checks both the parent and the mixture catalogue before
training. Each run initializes a fresh AdamW optimizer.

To rescore the restored endpoints, make the recorded training
JSON files available beside the archived weights.

```bash
study=checkpoints/residual-scale-study
mkdir -p "$study/results"
cp results/event_lm/residual_scale/*_s9631_residual_scale_*.json "$study/results/"
for arm in enwik8-snn wide-snn snn; do
  python experiments/probe_event_state.py --device cpu --enwik8 data/enwik8 \
    --chunk-size 512 --targets 65536 --checkpoint "$study/${arm}_s9631.pt" \
    --json "$study/results/probe_${arm}_s9631.json"
done
for arm in enwik8-snn snn wide-snn transformer wide-transformer; do
  bash experiments/run_v4_continuation.sh "residual-scale-$arm" "$study"
done

comparison=("$study"/{wide-transformer,transformer,snn,wide-snn,enwik8-snn}_s9631.pt)
for split in val test; do
  python experiments/probe_early_context.py --enwik8 data/enwik8 --device cpu --split "$split" \
    --checkpoint "${comparison[@]}" --width-gap "$study"/{wide-snn,wide-transformer,snn,transformer}_s9631.pt \
    --json "$study/results/early_context_$split.json"
done
python experiments/eval_byte_transfer.py --device mps --batch-size 24 --prepared data/transfer-v1/prepared \
  --checkpoint "${comparison[@]}" --json "$study/results/byte_transfer.json"
```

The fixed-weight 65,536-target probes above use CPU, and the
full-corpus scores and native parity audits use Metal for the
tensor forward pass. The early enwik8, enwik9 and C4 ablations
below run independently.

```bash
study=checkpoints/residual-scale-study
snn_pair=("$study"/{snn,wide-snn,enwik8-snn}_s9631.pt)
python experiments/probe_early_context.py --enwik8 data/enwik8 --device cpu --split val \
  --checkpoint "${snn_pair[@]}" --ablate-residual-stages \
  --json "$study/results/residual_stage_validation.json"
python experiments/probe_residual_transfer.py --prepared data/transfer-v1/prepared --corpus c4 --device cpu \
  --checkpoint "${snn_pair[@]}" --json "$study/results/residual_c4_validation.json"
python experiments/probe_residual_transfer.py --prepared data/transfer-v1/prepared --corpus enwik9 --device cpu \
  --checkpoint "$study"/{wide-snn,enwik8-snn}_s9631.pt --json "$study/results/residual_enwik9.json"
python experiments/probe_residual_history.py --checkpoint "$study/wide-snn_s9631.pt" \
  --enwik8 data/enwik8 --early-record "$study/results/early_context_val.json" \
  --prepared data/transfer-v1/prepared --json "$study/results/residual_history_validation.json"
```

## Regenerate the reports

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v10 --release-dir checkpoints/release-v10
```

The [leaderboard](LEADERBOARD.md) shows the two matched widths,
the corpus control, and the fixed transfer suite. The independent
corpus files can be reconstructed with the [source manifest](../results/event_lm/training_mixture/sources_v1.json)
and the [preparation code](../experiments/prepare_byte_training_mixture.py).
