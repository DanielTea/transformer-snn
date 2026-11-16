# ShuntFormer v3 research handoff

Start with the bundled five-block checkpoint after its 2,048-byte
continuation. Pick up an experimental branch from the checkpoint
archive. Run the shell examples from the repository root in Bash.
[Model card](../models/spiking-byte-lm-v3/README.md) ·
[Leaderboards](LEADERBOARD.md) · [Experiment status](#experiment-status)

## Set up a working checkout

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The v3 runs used Python 3.12.8, PyTorch 2.15.0 and NumPy 2.5.4 on an
Apple M3 Max with 48 GiB of unified memory. The training and scoring
commands below use Metal with CPU fallback disabled. Use `--device cpu`
for evaluation on CPU. The released plots use Matplotlib 3.11.2 and the
test suite uses pytest 9.2.0.

### Checkpoints

The checkout ships the seed-711 2,048-byte held-charge continuation
as `models/spiking-byte-lm-v3/weights.pt` and its seed-0 parent
under `models/spiking-byte-lm-v2/`. The v3 archive adds the paired
continuations, scaling runs, replication checkpoints, and the
Transformer parents. Each checkpoint has a SHA-256 and a training
record in the manifest.
[v3 release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v3) ·
[v3 manifest](../models/spiking-byte-lm-v3/manifest.json) ·
[Earlier depth and routing weights](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v2)

Download and extract the archive to restore the study paths under
`checkpoints/`.

```bash
mkdir -p checkpoints/downloads
gh release download spiking-byte-lm-v3 --repo DanielTea/transformer-snn \
  --pattern 'spiking-byte-lm-v3-checkpoints.*' --dir checkpoints/downloads
(
  cd checkpoints/downloads
  shasum -a 256 -c spiking-byte-lm-v3-checkpoints.sha256
)
tar -xzf checkpoints/downloads/spiking-byte-lm-v3-checkpoints.tar.gz
shasum -a 256 -c checkpoints/SHA256SUMS-v3
```

The seven-block checkpoint is `checkpoints/scale-3260k-study/trace_s0.pt`.
Use it with the same integer generator as the default. It has the
lowest PG-19 loss in the release and uses 31,692 bytes of stream state.
[Seven-block evaluation](SCALE_3260K.md)

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
  results/event_lm/long_context_continuation/event_correction_code_s711_ctx2048_continuation.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/replay-v3

python experiments/replay_event_run.py \
  results/event_lm/long_context_continuation/transformer_s711_ctx2048_continuation.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/replay-v3-transformer
```

For a new experiment initialized from v3, use the same spike geometry
and the chronological held-charge gradient. The checkpoint contains
weights and metadata. Each continuation starts a fresh AdamW optimizer
and scheduler. Record the new experiment under its own tag and output
path. To repeat a measured result, use its training record above.

```bash
spiking=(--variant event_correction_code --dim 264 --hidden 185 --depth 5
  --no-carry --memory-heads 4 --memory-slots 14 --memory-width 32
  --forecast-key-channels 2 --match-forecast-key --learn-context-keys
  --spike-resolution 16 --pulse-budget 310000 --code-loss-weight 0.25
  --compact-delays --activity-inhibition)
python experiments/train_event_lm.py "${spiking[@]}" --data data/enwik8 --device mps \
  --local-trace --trace-write-credit --initialize-from models/spiking-byte-lm-v3/weights.pt \
  --ctx 2048 --bs 2 --eval-bs 2 --steps 400 --warmup 32 --lr 0.0002 --seed 42 --validation-only \
  --tag continuation --results-dir checkpoints/work/results --checkpoint checkpoints/work/continuation.pt
```

The original v2 release carries the inhibition-only parent for the
paired delay-filter runs. Its filter and control training records use
the same replay runner. [Delay-filter study](DELAY_FILTER.md)

```bash
python experiments/replay_event_run.py \
  results/event_lm/delay_filter/event_correction_code_s610_warm_delay_filter.json \
  --enwik8 data/enwik8 --device mps \
  --output-dir checkpoints/work/delay-filter
```

## Score and audit

The SNN carries its charge and axon state through 8,192-byte evaluation
chunks. The 2,048-byte Transformer scores 1,024-byte chunks with up
to 1,024 preceding warmup bytes. Each scores 4,999,999 enwik8 test
targets and 8,404,368 PG-19 test targets. enwik8 uses the original
output scale. For PG-19 the evaluator selects one temperature per
checkpoint on the validation books and reports calibrated and raw loss.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work/scores
snn=models/spiking-byte-lm-v3/weights.pt
full=checkpoints/long-context-continuation/transformer_ctx2048_s711.pt
snn_score=(python experiments/eval_event_lm.py --device mps --context 8192 --history 256 --batch-size 8)
full_score=(python experiments/eval_event_lm.py --device mps --context 1024 --history 1024 --batch-size 4)
pg19=(--books-dir data/pg19-test
  --manifest results/event_lm/correction_pg19_full_book_manifest.json
  --calibration-books data/pg19-validation
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json
  --temperatures 0.8 1.0 1.15 1.25 1.4 1.5 1.65 1.8 2.0 2.25)

"${snn_score[@]}" --checkpoint "$snn" --enwik8 data/enwik8 \
  --json checkpoints/work/scores/enwik8_snn.json
"${full_score[@]}" --checkpoint "$full" --enwik8 data/enwik8 \
  --json checkpoints/work/scores/enwik8_full.json
"${snn_score[@]}" --checkpoint "$snn" "${pg19[@]}" \
  --json checkpoints/work/scores/pg19_snn.json
"${full_score[@]}" --checkpoint "$full" "${pg19[@]}" \
  --json checkpoints/work/scores/pg19_full.json

python experiments/audit_spike_code.py --device mps --split test --activity-profile \
  --enwik8 data/enwik8 --checkpoint "$snn" --json checkpoints/work/scores/native_snn.json
```

For the 512-byte Transformer runs, set `--context 256 --history 256
--batch-size 8`. Keep the SNN streaming settings for either training
length. Score files record the data digests, temperature grid, number
of targets, and model path used for each run.

## Experiment status

| Experiment | State at handoff | Evidence |
|---|---|---|
| Original held charge and inhibition | Depth controls and inhibition-only replication complete | [Depth study](DEPTH_INHIBITION.md) |
| Held-charge replication | Seeds 0–2 scored and audited | [Per-seed results](HELD_CHARGE_REPLICATION.md) |
| Opponent steering | 512-byte continuation and matched held-charge control scored and audited | [Circuit and intervention](OPPONENT_STEERING.md) |
| Chronological write credit | Matched continuation and forward/backward timing complete | [Training rule and timing](TRACE_WRITE_CREDIT.md) |
| Read veto | Matched continuation and gain-zeroing probe complete | [Circuit and intervention](TRACE_READ_VETO.md) |
| Spread coincidence | Matched continuation and timing-fraction-zeroing probe complete | [Circuit and intervention](SPREAD_COINCIDENCE.md) |
| 2,048-byte continuations | Both models scored on both corpora, and the SNN passed the integer audit | [Context comparison](CONTEXT_LENGTH.md) |
| Seven-block scaling study | Matched 3.26M runs and integer audit | [Scale comparison](SCALE_3260K.md) |

The archived 3.26M pair was trained on 512-byte windows. A matched
2,048-byte continuation at this size remains open. The release
includes both parent checkpoints, the 3.26M geometry, and the
1.63M continuation protocol.
[Scale configuration](#scale-to-326m) · [Longer-context protocol](CONTEXT_LENGTH.md#train-the-longer-context)

### Repeat the held-charge comparison

Use the `spiking` array from the warm-start example to rerun seeds
1 and 2 from initialization. The archived SNN and Transformer
checkpoints for all three seeds can be scored directly. Each seed
uses the same data and training budget as its reference.

```bash
common=(--data data/enwik8 --device mps --ctx 512 --bs 8 --steps 6400
  --lr 0.003 --warmup 160 --eval-every 400 --eval-bs 8 --eval-batches 8
  --final-eval-batches 48 --validation-only)
replication=checkpoints/work/held-charge-replication
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
disabled. The full-attention reference has 3,256,832 parameters.
Both runs used 26,214,400 enwik8 targets and 512-byte training
windows. Use fresh output paths to rerun them.
[MPS smoke record](../results/event_lm/scale_3260k/geometry_smoke.json) · [Scale results](SCALE_3260K.md)

```bash
scale=checkpoints/work/scale-3260k
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
| [Decoder](../snnformer/event_lm.py) and [charge updates](../snnformer/correction_memory.py) | Synapses, firing, delay groups, held charge, correction bank and integer export |
| [Transformer reference](../snnformer/transformer_reference.py) | Matched causal attention baseline |
| [Trainer](../experiments/train_event_lm.py) and [replay runner](../experiments/replay_event_run.py) | Shared training entry point and exact command reconstruction |
| [Evaluator](../experiments/eval_event_lm.py), [native audit](../experiments/audit_spike_code.py) and [state probe](../experiments/probe_event_state.py) | Full-corpus scores, integer parity, pulse activity and mechanism ablations |
| [Charge profile](../experiments/profile_charge_trace.py) and [training timing](../experiments/benchmark_trace_training.py) | Age of delivered charge and matched backward-pass measurements |
| [Tests](../tests) | Spike arithmetic, causality, integer parity and memory tests |
| [results](../results/README.md) | Raw JSON by study, per-book and per-seed results, manifests and figures |
| [v3 model](../models/spiking-byte-lm-v3) | Reference weights, architecture and release inventory |

Rebuild the tables, figures, and browser report directly from the tracked
records. Rebuild the optional checkpoint bundle from the restored
study directories. The packaging script verifies the per-file digests
and preserves the original checkpoint paths.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v3 --release-dir checkpoints/release-v3
```
