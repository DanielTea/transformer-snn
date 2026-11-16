# ShuntFormer v9 research handoff

The bundled checkpoint is the 14.07M unmatched-arrival SNN.
The release restores the four matched continuations and both
116,326,400-target parents. Training records include every
argument, configuration, seed and parent digest.
[Model and circuit](../models/spiking-byte-lm-v9/README.md) ·
[Arrival study](FIXED_ARRIVALS.md) · [Benchmarks](LEADERBOARD.md)

## Restore and run

Use the existing Python environment or set up Python 3.12 and
install `requirements.txt`. This study used Python 3.12.8 and
PyTorch 2.15.0 with Metal training and CPU integer audits.
Prepare canonical enwik8 and the
pinned PG-19 books with the earlier handoff. The byte transfer
suite uses the additional benchmark requirements and its pinned
source manifest.
[enwik8 and PG-19 data](HANDOFF.md#datasets) · [Transfer data](BYTE_TRANSFER.md#reproduce)

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v9
  for extension in sha256 tar.gz; do
    asset="spiking-byte-lm-v9-checkpoints.$extension"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
  done
  (cd checkpoints/downloads && shasum -a 256 -c spiking-byte-lm-v9-checkpoints.sha256)
  tar -xzf checkpoints/downloads/spiking-byte-lm-v9-checkpoints.tar.gz
  shasum -a 256 -c checkpoints/SHA256SUMS-v9
)
python experiments/generate_event_lm.py models/spiking-byte-lm-v9/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

The archive verifies each checkpoint by SHA-256. The three SNNs
and full-attention reference are `{control,selected,unmatched,transformer}_s9349.pt`
under `checkpoints/trace-arrival-study`. Their parents are
`{control,transformer}_s8237.pt` under
`checkpoints/trace-read-drive-study`.
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v9) ·
[Exact manifest](../models/spiking-byte-lm-v9/manifest.json)

## Replay the experiment

Use a fresh directory to retrain. The replay command verifies the
parent digest before launching from the recorded arguments. Each
continuation starts a fresh AdamW state and schedule.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
record=results/event_lm/trace_arrival/event_correction_code_s9349_trace_arrival_unmatched.json
python experiments/replay_event_run.py "$record" --enwik8 data/enwik8 \
  --device mps --output-dir checkpoints/work/replay-v9-unmatched
```

The study launcher trains and scores each arm. Set `SNN_TRAIN_ONLY=1`
to save the endpoint before running the corpus evaluations. To
score restored endpoints, copy their exact training JSON files
under the study's `results/` directory before invoking the launcher.

```bash
study=checkpoints/trace-arrival-study
mkdir -p "$study/results"
cp results/event_lm/trace_arrival/*_s9349_trace_arrival_*.json "$study/results/"
for arm in control selected unmatched transformer; do
  bash experiments/run_v4_continuation.sh "trace-arrival-$arm" "$study"
done

for split in val test; do
  python experiments/probe_early_context.py --enwik8 data/enwik8 --device cpu --split "$split" \
    --checkpoint "$study"/{transformer,control,selected,unmatched}_s9349.pt \
    --json "$study/results/early_context_$split.json"
done
python experiments/eval_byte_transfer.py --device mps --prepared data/transfer-v1/prepared \
  --checkpoint "$study"/{transformer,control,selected,unmatched}_s9349.pt \
  --json "$study/results/byte_transfer.json"
```

The same byte transfer suite applies to subsequent releases.
The source manifest and the prepared dataset catalogue keep
window selection and overlap decisions reproducible. Dataset
files live in the local data cache.
[Transfer setup, metrics and provenance](BYTE_TRANSFER.md)

## Verify the warm starts

The two routing projections preserve the parent predictions while
their weights are zero. The full-attention control preserves its
logits with the extra SwiGLU units' output weights at zero. Both
checks use validation bytes and a real training sample for gradient
flow.

```bash
parents=checkpoints/trace-read-drive-study
audit=checkpoints/work/v9-audits
mkdir -p "$audit"
python experiments/probe_early_context.py --enwik8 data/enwik8 --device cpu --split val \
  --profile-arrivals --checkpoint "$parents"/{transformer,control}_s8237.pt \
  --json "$audit/parent_arrival_profile.json"
for arm in selected unmatched; do
  python experiments/audit_residual_growth.py --device mps --enwik8 data/enwik8 \
    --parent "$parents/control_s8237.pt" --trace-arrival "$arm" --trace-arrival-groups 2 \
    --json "$audit/warm_$arm.json"
done
python experiments/audit_transformer_growth.py --device mps --enwik8 data/enwik8 \
  --parent "$parents/transformer_s8237.pt" --mlp-ratio 15.796875 --seed 9349 \
  --json "$audit/warm_transformer.json"
```

## Regenerate the release

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v9 --release-dir checkpoints/release-v9
```

## Source map

| Purpose | Source |
|---|---|
| Spike routing, held charge, fixed-arrival coincidence and native decoder | [snnformer/event_lm.py](../snnformer/event_lm.py) |
| Tagged error storage and inhibitory recall | [snnformer/correction_memory.py](../snnformer/correction_memory.py) |
| Training, warm-start growth and checkpoint replay | [train_event_lm.py](../experiments/train_event_lm.py) · [replay_event_run.py](../experiments/replay_event_run.py) |
| Standard corpora and paired early-context probe | [run_v4_continuation.sh](../experiments/run_v4_continuation.sh) · [eval_event_lm.py](../experiments/eval_event_lm.py) · [probe_early_context.py](../experiments/probe_early_context.py) |
| Transfer dataset preparation and scoring | [prepare_byte_benchmarks.py](../experiments/prepare_byte_benchmarks.py) · [eval_byte_transfer.py](../experiments/eval_byte_transfer.py) |
| Graphics, report and release bundle | [plot_event_lm_study.py](../experiments/plot_event_lm_study.py) · [make_report.py](../experiments/make_report.py) · [package_event_lm.py](../experiments/package_event_lm.py) |
