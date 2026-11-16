# ShuntFormer v2 research handoff

The bundled reference uses five spiking blocks with activity inhibition
and held charge. Run the commands from the repository root in Bash.
[Model card](../models/spiking-byte-lm-v2/README.md) ·
[Depth study](DEPTH_INHIBITION.md) · [Experiment status](#experiment-status)

## Set up a working checkout

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The v2 runs used Python 3.12.8, PyTorch 2.15.0 and NumPy 2.5.4 on an
Apple M3 Max with 48 GiB of unified memory. The training and scoring
commands below use Metal with CPU fallback disabled. Use `--device cpu`
for evaluation on CPU. The released plots use Matplotlib 3.11.2.

### Checkpoints

The repository ships the seed-0 held-charge weights. The release archive
includes training checkpoints for the depth, router, and matched
Transformer studies and their warm-start parents. Each checkpoint has
a SHA-256 digest and a link to its training record.
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v2) ·
[Manifest](../models/spiking-byte-lm-v2/manifest.json)

In a fresh checkout, download and extract the archive to restore the
original study subdirectories under `checkpoints/`.

```bash
mkdir -p checkpoints/downloads
gh release download spiking-byte-lm-v2 --repo DanielTea/transformer-snn \
  --pattern 'spiking-byte-lm-v2-checkpoints.*' --dir checkpoints/downloads
(
  cd checkpoints/downloads
  shasum -a 256 -c spiking-byte-lm-v2-checkpoints.sha256
)
tar -xzf checkpoints/downloads/spiking-byte-lm-v2-checkpoints.tar.gz
shasum -a 256 -c checkpoints/SHA256SUMS
```

### Datasets

Put the raw enwik8 file at `data/enwik8`. Its verified SHA-1 is
`57b8363b814821dc9d47aa4d41f58733519076b2`. The first 90,000,000 bytes
are training data, the next 5,000,000 are for validation, and the last
5,000,000 are for test. The PG-19 preparation commands download only
the text pinned by each content-addressed manifest. The test uses
16 complete books, and the temperature fit uses eight separate books.
[enwik8 source](https://www.mattmahoney.net/dc/textdata.html) ·
[PG-19 preparation](../experiments/prepare_pg19.py)

```bash
mkdir -p data
curl -fL https://www.mattmahoney.net/dc/enwik8.zip -o data/enwik8.zip
unzip -n -j data/enwik8.zip enwik8 -d data
python experiments/prepare_pg19.py \
  --manifest results/event_lm/correction_pg19_full_book_manifest.json --output data/pg19-test
python experiments/prepare_pg19.py \
  --manifest results/event_lm/correction_pg19_validation_manifest.json --output data/pg19-validation
```

## Train and continue

The training records store every CLI argument, architecture field, seed,
validation measurement, dataset checksum, and parent digest where
applicable. Replay with a new output directory. Use `--dry-run` to
inspect the exact generated command.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
python experiments/replay_event_run.py \
  results/event_lm/depth_inhibition/event_correction_code_s0_deep_trace.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/depth

python experiments/replay_event_run.py \
  results/event_lm/scale_1630k/transformer_s0_scale_1630k.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/full-attention
```

For a warm start from the bundled reference, use the v2 spike geometry,
activity inhibition, and the local charge trace. The archived training files
contain model weights and metadata. Continuation starts a fresh AdamW
optimizer and scheduler with the supplied seed and learning rate.

```bash
spiking=(--variant event_correction_code --dim 264 --hidden 185 --depth 5
  --no-carry --memory-heads 4 --memory-slots 14 --memory-width 32
  --forecast-key-channels 2 --match-forecast-key --learn-context-keys
  --spike-resolution 16 --pulse-budget 310000 --code-loss-weight 0.25
  --compact-delays --activity-inhibition)
python experiments/train_event_lm.py "${spiking[@]}" --data data/enwik8 --device mps \
  --local-trace --initialize-from models/spiking-byte-lm-v2/weights.pt \
  --ctx 512 --bs 8 --steps 400 --warmup 32 --lr 0.0002 --seed 42 --validation-only \
  --tag continuation --results-dir checkpoints/work/results --checkpoint checkpoints/work/continuation.pt
```

The paired delay-filter run uses the inhibition-only parent in the archive.
Swap the record suffix from `filter` to `control` for its matched
continuation.

```bash
python experiments/replay_event_run.py \
  results/event_lm/delay_filter/event_correction_code_s610_warm_delay_filter.json \
  --enwik8 data/enwik8 --device mps --parent-checkpoints checkpoints/depth-inhibition-study \
  --output-dir checkpoints/work/delay-filter
```

## Score and audit

The SNN carries its correction charge across each book. The
baseline finite-context evaluation scores 256-byte chunks with
256 preceding warmup bytes, so full attention sees at most 512 bytes.
Both score all 4,999,999 enwik8 test targets and all 8,404,368 PG-19 test
targets. enwik8 uses the original output scale. For PG-19 the evaluator
selects one temperature per checkpoint on the validation books and
reports calibrated and uncalibrated loss.

```bash
mkdir -p checkpoints/work/scores
snn=models/spiking-byte-lm-v2/weights.pt
full=checkpoints/scale-1630k-study/transformer_s0.pt
score=(python experiments/eval_event_lm.py --device mps --history 256 --batch-size 8)
pg19=(--books-dir data/pg19-test
  --manifest results/event_lm/correction_pg19_full_book_manifest.json
  --calibration-books data/pg19-validation
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json
  --temperatures 0.8 1.0 1.15 1.25 1.4 1.5 1.65 1.8 2.0 2.25)

"${score[@]}" --context 8192 --checkpoint "$snn" --enwik8 data/enwik8 \
  --json checkpoints/work/scores/enwik8_snn.json
"${score[@]}" --context 256 --checkpoint "$full" --enwik8 data/enwik8 \
  --json checkpoints/work/scores/enwik8_full.json
"${score[@]}" --context 8192 --checkpoint "$snn" "${pg19[@]}" \
  --json checkpoints/work/scores/pg19_snn.json
"${score[@]}" --context 256 --checkpoint "$full" "${pg19[@]}" \
  --json checkpoints/work/scores/pg19_full.json

python experiments/audit_spike_code.py --device mps --split test --activity-profile \
  --enwik8 data/enwik8 --checkpoint "$snn" --json checkpoints/work/scores/native_snn.json
```

## Experiment status

| Experiment | State at handoff | Evidence |
|---|---|---|
| Five blocks with activity inhibition, seed 0 | Complete on both corpora and audited in integer CPU inference | [Depth study](DEPTH_INHIBITION.md) |
| Five blocks with held charge | Complete on both corpora in seed 0. Native per-byte and chunk parity verified. | [Charge rule and evaluation](DEPTH_INHIBITION.md#charge-held-by-steering-spikes) |
| Inhibition-only replication | Both corpora and models are scored for seeds 0–2 | [Per-seed results](DEPTH_INHIBITION.md#full-attention-replication) |
| Paired delay filter, seed 610 | Both arms trained for an additional 4,915,200 targets from the same checkpoint | [Filter design and measurements](DELAY_FILTER.md) |
| Seven-block scaling study | MPS forward and backward verified on a full training batch. Training and corpus comparisons are pending. | [Scale configuration](#scale-to-326m) |

### Repeat the held-charge comparison

Run seeds 1 and 2 with the `spiking` array from the warm-start example
and a fresh initialization. The matched full-attention checkpoints are
already in the archive. Each seed uses the same data and training
budget as its reference. Use the resulting checkpoints in the SNN
scoring commands above.

```bash
common=(--data data/enwik8 --device mps --ctx 512 --bs 8 --steps 6400
  --lr 0.003 --warmup 160 --eval-every 400 --eval-bs 8 --eval-batches 8
  --final-eval-batches 48 --validation-only)
replication=checkpoints/held-charge-replication
for seed in 1 2; do
  python experiments/train_event_lm.py "${common[@]}" "${spiking[@]}" \
    --local-trace --seed "$seed" --tag held_charge_replication \
    --results-dir "$replication/results" --checkpoint "$replication/trace_s$seed.pt"
done
```

### Scale to 3.26M

The seven-block geometry uses 384 residual channels and 187 expansion
channels per block. It has 3,257,088 parameters and 31,692 bytes of
persistent state with `--local-trace`. The inhibition-only control
uses 26,456 bytes. Both completed a forward and
backward pass on eight real 512-byte enwik8 windows with MPS fallback
disabled. The proposed full-attention reference has 3,256,832 parameters
and a 512-byte training context. Keep training exposure fixed for the
first scale comparison. Compare the full-corpus losses and record
measured pulse counts.
[MPS smoke record](../results/event_lm/scale_3260k/geometry_smoke.json)

```bash
scale=checkpoints/scale-3260k-study
python experiments/train_event_lm.py "${common[@]}" "${spiking[@]}" \
  --dim 384 --hidden 187 --depth 7 --seed 0 --local-trace --tag scale_3260k_trace \
  --results-dir "$scale/results" --checkpoint "$scale/trace_s0.pt"
python experiments/train_event_lm.py "${common[@]}" --variant transformer \
  --dim 256 --depth 4 --heads 4 --mlp-ratio 4.078125 --seed 0 --tag scale_3260k \
  --results-dir "$scale/results" --checkpoint "$scale/transformer_s0.pt"
```

## Source and release map

| Location | Contents |
|---|---|
| [snnformer](../snnformer) | Decoder, integer export, memory state and Transformer reference |
| [experiments](../experiments) | Training, replay, corpus preparation, scoring, interventions and figures |
| [tests](../tests) | Spike arithmetic, causality, integer parity and memory tests |
| [results](../results/README.md) | Raw JSON by study, per-book and per-seed results, manifests and figures |
| [v2 model](../models/spiking-byte-lm-v2) | Reference weights, architecture and release inventory |

Rebuild the tables, figures, and browser report directly from the tracked
records. Rebuild the optional checkpoint bundle from the restored
study directories. The packaging script verifies the per-file digests
and preserves the original checkpoint paths.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v2 --release-dir checkpoints/release-v2
```
