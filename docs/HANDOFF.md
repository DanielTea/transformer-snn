# ShuntFormer v6 research handoff

The bundled 11-block SNN is the depth-growth endpoint at 81.92M
cumulative enwik8 training targets. The v6 archive restores the six
matched endpoints, the preceding width and circuit comparisons,
and their parent checkpoints. Run these commands from the repository
root in Bash.
[Model card](../models/spiking-byte-lm-v6/README.md) ·
[Leaderboards](LEADERBOARD.md) · [Continue from v6](#continue-from-the-v6-endpoints)

## Set up a working checkout

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The experiments used Python 3.12.8, PyTorch 2.15.0 and NumPy 2.5.4 on an
Apple M3 Max with 48 GiB of unified memory. The training and scoring
commands below use Metal with CPU fallback disabled. Use `--device cpu`
for evaluation on CPU. The released plots use Matplotlib 3.11.2 and the
test suite uses pytest 9.2.0.

### Checkpoints

The reference weights live in `models/spiking-byte-lm-v6/weights.pt`.
The manifest links all 23 archive checkpoints to their training records,
parent paths, and SHA-256 digests. Extract from the repository root to restore
the paths used by the training and scoring scripts.
[v6 release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v6) ·
[v6 manifest](../models/spiking-byte-lm-v6/manifest.json)

```bash
mkdir -p checkpoints/downloads
gh release download spiking-byte-lm-v6 --repo DanielTea/transformer-snn \
  --pattern 'spiking-byte-lm-v6-checkpoints.*' --dir checkpoints/downloads
(
  cd checkpoints/downloads
  shasum -a 256 -c spiking-byte-lm-v6-checkpoints.sha256
)
tar -xzf checkpoints/downloads/spiking-byte-lm-v6-checkpoints.tar.gz
shasum -a 256 -c checkpoints/SHA256SUMS-v6
```

The six current endpoints are under `checkpoints/depth-growth-study/`.

| Checkpoint | Purpose |
|---|---|
| `depth_s4603.pt` | Bundled 11-block reference, signed corrections in blocks 8–11 |
| `wide_s4603.pt` | 1,174-channel SNN with seven blocks and the matched 9.96M parameter count |
| `transformer_s4603.pt` | Six-block, 9.96M full-attention reference |
| `control_s4603.pt` | Seven-block, 768-channel SNN with ordinary residual output |
| `corrections_s4603.pt` | Same seven-block geometry with signed corrections at every output |
| `control_transformer_s4603.pt` | Four-block, 6.51M full-attention reference |

The v5 archive carries additional expansion, write confirmation,
inhibition, and teacher-guidance checkpoints. Earlier releases carry
the original routing and context-tag studies.
[v5 assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v5) ·
[v5 manifest](../models/spiking-byte-lm-v5/manifest.json) ·
[Earlier releases](https://github.com/DanielTea/transformer-snn/releases)

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
printf '57b8363b814821dc9d47aa4d41f58733519076b2  data/enwik8\n' | shasum -a 1 -c
python experiments/prepare_pg19.py \
  --manifest results/event_lm/correction_pg19_full_book_manifest.json --output data/pg19-test
python experiments/prepare_pg19.py \
  --manifest results/event_lm/correction_pg19_validation_manifest.json --output data/pg19-validation
```

## Continue from the v6 endpoints

Use the archived weights as `--initialize-from` inputs for the
next matched continuation. Keep the 8-bit local synapses, 16×
firing-count resolution, and 310,000-pulse training target for the
depth and width arms. The 11-block reference uses
`--residual-correction-start 7`, with zero-based indexing. Supply
`--grow-depth` when adding blocks. Under the existing tap schedule
a twelfth block uses 34,128 B, so a further depth increase first
needs a change to the delay-state layout or budget.
[Depth circuit and ablations](DEPTH_GROWTH.md)

The commands below launch a fresh matched exposure study from the
81.92M v6 endpoints, using a shared seed, 2,400 updates and the same
512-byte windows. Choose separate output paths for other designs.
Each exported checkpoint stores only the model state needed to start
a fresh continuation. AdamW and the learning-rate schedule start
fresh on each invocation. A one-update CPU check on enwik8 verified
that all three recipes load the released endpoints and can train.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
parent=checkpoints/depth-growth-study
next=checkpoints/work/next-exposure
shared=(--data data/enwik8 --device mps --ctx 512 --bs 8 --steps 2400
  --warmup 96 --lr 0.0002 --seed 5801 --eval-every 300 --eval-bs 8
  --eval-batches 8 --final-eval-batches 48 --results-dir "$next/results"
  --validation-only)
snn=(--variant event_correction_code --hidden 187 --no-carry --memory-heads 4
  --memory-slots 14 --memory-width 32 --forecast-key-channels 2
  --match-forecast-key --learn-context-keys --spike-resolution 16
  --pulse-budget 310000 --code-loss-weight 0.25 --router-charge-bits 8
  --compact-delays --compact-axon-state --activity-inhibition
  --inhibition-by-expansion --local-trace --trace-write-credit)

python experiments/train_event_lm.py "${shared[@]}" "${snn[@]}" \
  --dim 768 --depth 11 --residual-correction-start 7 \
  --initialize-from "$parent/depth_s4603.pt" --tag next_exposure_depth \
  --checkpoint "$next/depth.pt"
python experiments/train_event_lm.py "${shared[@]}" "${snn[@]}" \
  --dim 1174 --depth 7 --initialize-from "$parent/wide_s4603.pt" \
  --tag next_exposure_width --checkpoint "$next/width.pt"
python experiments/train_event_lm.py "${shared[@]}" \
  --variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 10.576171875 \
  --initialize-from "$parent/transformer_s4603.pt" \
  --tag next_exposure_full --checkpoint "$next/full.pt"
```

## Replay the released runs

The training records store every CLI argument, architecture field, seed,
validation measurement, dataset checksum, and parent digest where
applicable. Replay with a new output directory. Use `--dry-run` to
inspect the exact generated command.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
python experiments/replay_event_run.py \
  results/event_lm/depth_growth/event_correction_code_s4603_depth_growth_snn.json \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/replay-v6-depth
```

Run the study launcher to train, score both corpora, and produce the
applicable native audits and causal interventions. The v6 run started
each arm from its 72.09M parent and added 9.83M targets.
[All six command lines, curves, scores, and audits](../results/event_lm/depth_growth)

```bash
for arm in control snn wide transformer corrections control-transformer; do
  bash experiments/run_v4_continuation.sh "depth-growth-$arm"
done
```

Use the corresponding training JSON for any other completed run.
The replay runner checks its parent digest before starting. An
interrupted training stage replays from its recorded starting
checkpoint. To run only scoring and audits on restored weights, use
the commands below.

### Resume the 13.03M comparison

Restore the v6 checkpoints and datasets above. The three arms start
from the two 65.54M checkpoints and each add 6,553,600 targets, using
seed 3307. The launcher trains each arm and scores both corpora. It
also runs the native parity and pulse audit for the SNN arms. With no
output argument it writes under `checkpoints/work/<study>`.
[Width design and warm starts](GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state) ·
[Training launcher](../experiments/run_v4_continuation.sh)

```bash
bash experiments/run_v4_continuation.sh residual-growth-snn
bash experiments/run_v4_continuation.sh residual-growth-transformer
bash experiments/run_v4_continuation.sh residual-growth-control
```

For the original-width Transformer comparison at the same exposure,
run `bash experiments/run_v4_continuation.sh residual-growth-control-transformer`.
[Held-out scores for both widths](GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state)

On the original workstation these arms use
`checkpoints/residual-13030k-study/`, with logs under `logs/` and
records under `results/`. The expected output weights are
`residual_s3307.pt`, `transformer_s3307.pt`, and `control_s3307.pt`.
Open the current log to see whether a local arm has finished. Reuse
that arm's output directory to finish missing scores or audits. The
launcher skips completed stages and preserves their reports. If a run
stopped after writing weights but before writing the training record,
use a fresh output directory to replay from the archived parent.

### Earlier exposure stages

The first pair below continues the grown models to 49,152,000
cumulative targets. The second pair continues from those endpoints
to 65,536,000. The final command replays the 32.77M growth test
with three-way coincidence in the added residual neurons.
[Exposure results](GROWN_RESIDUAL.md#extend-training-exposure) ·
[Three-way coincidence](TRACE_COINCIDENCE.md)

```bash
bash experiments/run_v4_continuation.sh extend-snn
bash experiments/run_v4_continuation.sh extend-transformer
bash experiments/run_v4_continuation.sh scale-data-snn
bash experiments/run_v4_continuation.sh scale-data-transformer
bash experiments/run_v4_continuation.sh trace-coincidence
```

The second pair reads the first pair's archived paths under
`checkpoints/scale-6510k-data-study/`. To train the full chain from
locally reproduced parents, use that directory as the output
argument for both `extend-*` commands in a fresh workspace, or use
the replay runner with a parent directory and its verified digests.

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
snn=models/spiking-byte-lm-v6/weights.pt
full=checkpoints/depth-growth-study/transformer_s4603.pt
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
| Expansion to 8.74M | Four arms scored. Audits confirm the restored steering population | [Expansion study](EXPANSION_GROWTH.md) |
| 13.03M residual growth | Both widths scored. The wider SNN improved PG-19 over the original-width control and full attention led on enwik8 | [Held-out comparison](GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state) |
| Tag read competition | All arms scored. The active-start runs read older charge and increased held-out loss on both corpora | [Read circuit and scores](TAG_READ_COMPETITION.md) |
| Late steering | Both corpora scored and late spikes measured. The original-width control led on both | [Feedback circuit and scores](LATE_STEERING.md) |
| Finer router charge at fixed width | Warm counts agree with v5. The finer router improved enwik8 and increased PG-19 loss in the matched continuation | [Precision study and scores](GROWN_RESIDUAL.md#refine-router-charge-at-fixed-width) |
| Depth growth with signed residual corrections | All six arms scored on both corpora. The deeper SNN, wider SNN and two seven-block SNNs passed the native audits | [Depth study](DEPTH_GROWTH.md) |
| Redistribution of inhibition | Matched continuation, both scores, native audit and transfer intervention | [Inhibition study](INHIBITION_REDISTRIBUTION.md) |
| Frozen teacher guidance | Equal-window control, held-out scores and native audit. The student uses extra training compute | [Training study](TEACHER_GUIDANCE.md) |
| Delayed write confirmation | Both corpora scored. Native activity and write age measured | [Write confirmation study](TRACE_WRITE_CONFIRM.md) |
| 65.54M cumulative-target extension | Matched pair scored on both corpora. The SNN is the v5 reference | [Exposure study](GROWN_RESIDUAL.md#extend-training-exposure) |
| Grown 6.51M matched pair | Trained and scored on both corpora, with SNN native parity and an added-neuron ablation | [Growth study](GROWN_RESIDUAL.md) |
| Grown 6.51M with finer charge | Released reference. Both corpora scored with native parity | [Finer-charge study](GROWN_RESIDUAL.md#finer-synaptic-charge-at-the-same-firing-resolution) |
| 6.51M pair from scratch | Matched 26.21M-target comparison scored and audited | [Width study](SCALE_6510K_WIDTH.md) |
| 3.26M context and memory variants | 512- and 2,048-byte continuations, signed residual membranes, four-phase address and corrected forecast scored | [Context](CONTEXT_LENGTH.md#repeat-at-326m-parameters) · [Membranes](RESIDUAL_MEMBRANE.md) · [Address phases](TAG_PHASE_ADDRESS.md) |
| Three-way coincidence in added neurons | Both scores and native audits complete. The matched ordinary SNN led both corpora | [Held-out comparison](TRACE_COINCIDENCE.md#paired-training-run) |
| 49.15M cumulative-target extension | Matched pair scored on both corpora and native SNN activity audited | [Continuation commands](#earlier-exposure-stages) · [Paired records](../results/event_lm/scale_6510k_data) |
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
| [SNN growth audit](../experiments/audit_residual_growth.py) and [Transformer and steering audits](../experiments/audit_growth_gradients.py) | Reproduce initialization parity and gradient checks on enwik8 |
| [Charge profile](../experiments/profile_charge_trace.py), [training timing](../experiments/benchmark_trace_training.py), and [decoder timing](../experiments/benchmark_decode.py) | Age of delivered charge, backward-pass timing, and native CPU decode measurements |
| [Tests](../tests) | Spike arithmetic, causality, integer parity and memory tests |
| [results](../results/README.md) | Raw JSON by study, per-book and per-seed results, manifests and figures |
| [v6 model](../models/spiking-byte-lm-v6) and [packager](../experiments/package_event_lm.py) | Reference weights, architecture and archive assembly |

Rebuild the tables, figures, and browser report directly from the tracked
records. Rebuild the optional checkpoint bundle from the restored
study directories. The packaging script verifies the per-file digests
and preserves the original checkpoint paths.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v6 --release-dir checkpoints/release-v6
```
