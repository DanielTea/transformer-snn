# ShuntFormer v4 research handoff

Start with the bundled seven-block, 6.51M-parameter SNN. The
checkpoint archive restores the exact 3.26M parents, grown pair,
and the other scored experiments. Run the shell examples from the
repository root in Bash.
[Model card](../models/spiking-byte-lm-v4/README.md) ·
[Leaderboards](LEADERBOARD.md) · [Experiment status](#experiment-status)

## Set up a working checkout

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The v4 runs used Python 3.12.8, PyTorch 2.15.0 and NumPy 2.5.4 on an
Apple M3 Max with 48 GiB of unified memory. The training and scoring
commands below use Metal with CPU fallback disabled. Use `--device cpu`
for evaluation on CPU. The released plots use Matplotlib 3.11.2 and the
test suite uses pytest 9.2.0.

### Checkpoints

The reference is `models/spiking-byte-lm-v4/weights.pt`. The v4
archive contains 29 checkpoints across the v3 set and the newly
completed studies. Each member is linked to its training record and
SHA-256 in the manifest.
[v4 release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v4) ·
[v4 manifest](../models/spiking-byte-lm-v4/manifest.json) ·
[Earlier depth and routing weights](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v2)

Download and extract the archive to restore the study paths under
`checkpoints/`.

```bash
mkdir -p checkpoints/downloads
gh release download spiking-byte-lm-v4 --repo DanielTea/transformer-snn \
  --pattern 'spiking-byte-lm-v4-checkpoints.*' --dir checkpoints/downloads
(
  cd checkpoints/downloads
  shasum -a 256 -c spiking-byte-lm-v4-checkpoints.sha256
)
tar -xzf checkpoints/downloads/spiking-byte-lm-v4-checkpoints.tar.gz
shasum -a 256 -c checkpoints/SHA256SUMS-v4
```

The reference is also archived as
`checkpoints/grown-fine-charge-study/fine_s711.pt`.
The six-bit grown control and its matched reference are
`checkpoints/grown-6510k-residual-study/grown_s711.pt`
and `checkpoints/grown-6510k-residual-study/transformer_s711.pt`.
The 3.26M SNN parent is `checkpoints/scale-3260k-study/trace_s0.pt`.
[Growth comparison](GROWN_RESIDUAL.md) · [Parent comparison](SCALE_3260K.md)

### Datasets

Put the raw enwik8 file at `data/enwik8`. Its verified SHA-1 is
`57b8363b814821dc9d47aa4d41f58733519076b2`. The first 90,000,000 bytes
are training data, the next 5,000,000 are for validation, and the last
5,000,000 are for test. The PG-19 preparation commands download only
the text pinned by each content-addressed manifest. The test uses
16 complete books, and the temperature fit uses eight separate books.
[enwik8 source](https://www.mattmahoney.net/dc/textdata.html) ·
[PG-19 source](https://github.com/google-deepmind/pg19) ·
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
  results/event_lm/grown_fine_charge/event_correction_code_s711_scale_6510k_grown_fine_charge.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/replay-v4

python experiments/replay_event_run.py \
  results/event_lm/grown_6510k_residual/transformer_s711_scale_6510k_grown_ffn.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/replay-v4-transformer
```

Use the corresponding training JSON for any other completed run.
Its parent digest is checked before training. Exported checkpoints
store weights, configuration, and training metadata. Each
continuation starts fresh AdamW and scheduler state. An interrupted
training stage must be replayed from its recorded starting checkpoint.

### Pick up the current experiments

The first two commands extend the six-bit grown SNN and its matched
Transformer by 16,384,000 enwik8 targets to 49,152,000 total.
The third repeats the original
32,768,000-target growth protocol with three-way coincidence in the
added residual neurons. Each command trains, scores both corpora, and
runs the integer audit for SNNs. Both 49.15M arms have full results.
The published v4 archive contains the two starting checkpoints and
the extended Transformer. On the original workstation, the extended
SNN is saved as `checkpoints/scale-6510k-data-study/trace_s1109.pt`.
[49.15M scores](GROWN_RESIDUAL.md#extend-training-exposure)
[Continuation script](../experiments/run_v4_continuation.sh) ·
[Three-way coincidence design](TRACE_COINCIDENCE.md)

```bash
bash experiments/run_v4_continuation.sh extend-snn
bash experiments/run_v4_continuation.sh extend-transformer
bash experiments/run_v4_continuation.sh trace-coincidence
```

### Shared options for earlier studies

The following profile and example reproduce the earlier v3
geometry. Several study notes reuse the `spiking` array and
override its width, depth, and expansion size.

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

## Score checkpoints

The SNN carries its charge and axon state through 8,192-byte evaluation
chunks and resets state between books. The 512-byte Transformer
scores 256-byte chunks with up to 256 preceding warmup bytes.
Each scores 4,999,999 enwik8 test targets and 8,404,368 PG-19 test
targets. enwik8 uses the original output scale. For PG-19 the
evaluator selects one temperature per checkpoint on the validation
books and reports calibrated and raw loss.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work/scores
snn=models/spiking-byte-lm-v4/weights.pt
full=checkpoints/grown-6510k-residual-study/transformer_s711.pt
snn_score=(python experiments/eval_event_lm.py --device mps --context 8192 --history 256 --batch-size 8)
full_score=(python experiments/eval_event_lm.py --device mps --context 256 --history 256 --batch-size 8)
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

For the 2,048-byte Transformer runs, use `--context 1024 --history 1024
--batch-size 4`. Use the same SNN streaming settings for either
training length. Score files record the data digests, temperature grid,
number of targets, and checkpoint path.

## Experiment status

| Experiment | State at handoff | Evidence |
|---|---|---|
| Expansion to 8.74M | Both wide arms and the original-width control scored. The steering-gradient rerun is queued | [Expansion study](EXPANSION_GROWTH.md) |
| Delayed write confirmation | Integer parity and full continuation training complete. Held-out scoring in progress | [Write confirmation study](TRACE_WRITE_CONFIRM.md) |
| Grown 6.51M matched pair | Trained and scored on both corpora, with SNN native parity and an added-neuron ablation | [Growth study](GROWN_RESIDUAL.md) |
| Grown 6.51M with finer charge | Released reference. Both corpora scored with native parity | [Finer-charge study](GROWN_RESIDUAL.md#finer-synaptic-charge-at-the-same-firing-resolution) |
| 6.51M pair from scratch | Matched 26.21M-target comparison scored and audited | [Width study](SCALE_6510K_WIDTH.md) |
| 3.26M context and memory variants | 512- and 2,048-byte continuations, signed residual membranes, four-phase address and corrected forecast scored | [Context](CONTEXT_LENGTH.md#repeat-at-326m-parameters) · [Membranes](RESIDUAL_MEMBRANE.md) · [Address phases](TAG_PHASE_ADDRESS.md) |
| Three-way coincidence in added neurons | Both scores and native audits complete. The matched ordinary SNN led both corpora | [Held-out comparison](TRACE_COINCIDENCE.md#paired-training-run) |
| 49.15M cumulative-target extension | Matched pair scored on both corpora and native SNN activity audited | [Continuation commands](#pick-up-the-current-experiments) · [Paired records](../results/event_lm/scale_6510k_data) |
| Original held charge and inhibition | Depth controls and inhibition-only replication complete | [Depth study](DEPTH_INHIBITION.md) |
| Held-charge replication | Seeds 0–2 scored and audited | [Per-seed results](HELD_CHARGE_REPLICATION.md) |
| Opponent steering | 512-byte continuation and matched held-charge control scored and audited | [Circuit and intervention](OPPONENT_STEERING.md) |
| Chronological write credit | Matched continuation and forward/backward timing complete | [Training rule and timing](TRACE_WRITE_CREDIT.md) |
| Read veto | Matched continuation and gain-zeroing probe complete | [Circuit and intervention](TRACE_READ_VETO.md) |
| Spread coincidence | Matched continuation and timing-fraction-zeroing probe complete | [Circuit and intervention](SPREAD_COINCIDENCE.md) |
| 2,048-byte continuations | Both models scored on both corpora, and the SNN passed the integer audit | [Context comparison](CONTEXT_LENGTH.md) |
| Seven-block scaling study | Matched 3.26M runs and integer audit | [Scale comparison](SCALE_3260K.md) |

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
| [Growth audit](../experiments/audit_residual_growth.py) | Reproduce initialization parity and gradient checks on enwik8 |
| [Charge profile](../experiments/profile_charge_trace.py) and [training timing](../experiments/benchmark_trace_training.py) | Age of delivered charge and matched backward-pass measurements |
| [Tests](../tests) | Spike arithmetic, causality, integer parity and memory tests |
| [results](../results/README.md) | Raw JSON by study, per-book and per-seed results, manifests and figures |
| [v4 model](../models/spiking-byte-lm-v4) | Reference weights, architecture and release inventory |

Rebuild the tables, figures, and browser report directly from the tracked
records. Rebuild the optional checkpoint bundle from the restored
study directories. The packaging script verifies the per-file digests
and preserves the original checkpoint paths.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v4 --release-dir checkpoints/release-v4
```
