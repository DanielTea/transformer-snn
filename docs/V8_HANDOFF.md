# ShuntFormer v8 research handoff

Start with the bundled 12.80M two-rail SNN for generation and
architecture work. Restore the archive to compare its controls or
continue training from the exact study endpoints. Both corpora were
scored on held-out text at a matched exposure of 116,326,400 enwik8
training targets using seed 8237 for the continuation.
[Model card and circuit](../models/spiking-byte-lm-v8/README.md) ·
[Leaderboard](LEADERBOARD.md) · [Read experiments](TRACE_READ_DRIVE.md)

## Set up

Run from the repository root in Bash. The release used Python 3.12.8,
PyTorch 2.15.0, NumPy 2.5.4 and Matplotlib 3.11.2 on an Apple M3 Max
with 48 GiB of unified memory. Generation uses the native CPU integer
decoder. The training recipes default to Metal with CPU fallback
disabled. Set `DEVICE=cpu` for a CPU launcher run or use the matching
`--device` option on the individual Python commands.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python experiments/generate_event_lm.py models/spiking-byte-lm-v8/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

Prepare `data/enwik8`, `data/pg19-test` and `data/pg19-validation`
with the canonical enwik8 checksum and pinned PG-19 manifests.
[Download and verify the datasets](HANDOFF.md#datasets)

## Restore checkpoints

The 20-checkpoint archive includes the v7 SNN and Transformer parents,
all seven packed-width runs, the read-drive endpoints and the wider SNN's
one-update smoke checkpoint. The manifest links each checkpoint to an
exact training record and parent digest. The bundled weight file and
`checkpoints/trace-read-drive-study/write_rails_share_s8237.pt` are identical.
[Release assets](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v8) ·
[Checkpoint manifest](../models/spiking-byte-lm-v8/manifest.json)

```bash
(
  set -e
  mkdir -p checkpoints/downloads
  release=https://github.com/DanielTea/transformer-snn/releases/download/spiking-byte-lm-v8
  for extension in sha256 tar.gz; do
    asset="spiking-byte-lm-v8-checkpoints.$extension"
    curl -fL --retry 3 "$release/$asset" -o "checkpoints/downloads/$asset"
  done
  (cd checkpoints/downloads && shasum -a 256 -c spiking-byte-lm-v8-checkpoints.sha256)
  tar -xzf checkpoints/downloads/spiking-byte-lm-v8-checkpoints.tar.gz
  shasum -a 256 -c checkpoints/SHA256SUMS-v8
)
```

| Path relative to the repository | When to use it |
|---|---|
| `checkpoints/trace-read-drive-study/write_rails_share_s8237.pt` | Bundled 12.80M SNN with independent write gains and shared steering |
| `checkpoints/trace-read-drive-study/write_carry_s8237.pt` | Signed write-carry reference and the parent used by the 25.6M scale recipe |
| `checkpoints/trace-read-drive-study/control_s8237.pt` | Same-exposure SNN with the ordinary held-charge read and 28,584 B of state |
| `checkpoints/trace-read-drive-study/transformer_s8237.pt` | Matched six-block full-attention reference |
| `checkpoints/trace-read-drive-study/*_s8237.pt` | Reset, carry, shared steering and separate-rail read comparisons |
| `checkpoints/packed-width-study/short_groups_s7907.pt` | Parent of the read-drive SNN arms at 111,411,200 targets |
| `checkpoints/packed-width-study/transformer_s7907.pt` | Parent of the read-drive Transformer at 111,411,200 targets |
| `checkpoints/trace-tap-study/{control,transformer}_s6899.pt` | Parents of the packed-width study at 101,580,800 targets |
| `checkpoints/trace-read-drive-study/scale-smoke/write_carry_wide_smoke_s9277.pt` | 25.6M SNN after one optimizer update on 4,096 targets |

## Replay a training run

Every archived training JSON stores the model configuration, CLI
arguments, seed, exposure and parent hash. Replay verifies the parent
hash before training. A new run or continuation creates a fresh
AdamW state and learning-rate schedule from the saved master weights.
Use a fresh output directory for each replay.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
record=results/event_lm/trace_read_drive/event_correction_code_s8237_trace_read_drive_write_rails_share.json
python experiments/replay_event_run.py "$record" \
  --enwik8 data/enwik8 --device mps --output-dir checkpoints/work/replay-v8-rails
```

The study launcher can train and score any arm from its restored parent.
The width runs add 9,830,400 targets with seed 7907. The read-drive runs
add 4,915,200 targets with seed 8237. `--dry-run` on the replay script
prints the corresponding command without starting training.
[Width recipes](PACKED_WIDTH.md#matched-runs) ·
[Read-drive recipes](TRACE_READ_DRIVE.md#match-inputs-training-and-evaluation)

## Evaluation

On a restored run, copy its checked-in training record beside the
weights before invoking the launcher. It uses the existing training
record and skips any report file that already exists. To regenerate
reports from scratch, place the checkpoint and training record
in a separate study directory.

```bash
study=checkpoints/work/rescore-v8-rails
record=results/event_lm/trace_read_drive/event_correction_code_s8237_trace_read_drive_write_rails_share.json
mkdir -p "$study/results"
cp checkpoints/trace-read-drive-study/write_rails_share_s8237.pt "$study/"
cp "$record" "$study/results/"
bash experiments/run_v4_continuation.sh trace-read-drive-write-rails-share "$study"
```

The scorer covers 4,999,999 enwik8 test targets and 8,404,368
targets from 16 full PG-19 test books. It carries SNN state
through 8,192-byte chunks and resets at book boundaries. The
Transformer scores 256-byte chunks with up to 256 warmup bytes.
The PG-19 report includes both raw loss and loss at a checkpoint's
temperature fitted on eight separate books. State counts persistent
tensors for one stream. Pulse totals count emitted events on a
4,096-byte native sample. The companion probe measures fixed-weight
interventions on 65,536 enwik8 validation targets.
[Evaluation flags and split checks](HANDOFF.md#score-checkpoints) ·
[Raw results by study](../results/README.md)

The context probes and generation samples for the earlier 12.8M
width comparison are reproducible with the archived paired windows.
The matched decode benchmark times CPU software at batch one.
[Probe results and commands](PACKED_WIDTH.md#paired-context-and-cpu-decoding)

## Resume with the 25.6M comparison

The scale-only recipe uses the signed write-carry checkpoint for
both SNN arms. Doubling its residual stream from 768 to 1,536
raises the SNN from 12,802,161 to 25,601,649 parameters at the same
31,257-byte state budget. The enlarged network reproduces every
parent count and pulse in the 4,096-byte warm-start audit. New
projection weights receive gradients in all eleven blocks. A Metal
smoke run completed one AdamW update on eight real 512-byte enwik8
training windows. The smoke checkpoint records that update.
[Warm SNN audit](../results/event_lm/trace_read_drive/warm_write_carry_residual_1536.json) ·
[One-update training record](../results/event_lm/trace_read_drive/event_correction_code_s9277_wide_read_training_smoke.json)

Growing the bundled two-rail SNN in the same way yields 25,604,322
parameters. Its warm start preserves all parent counts and pulses
at 31,257 bytes of state and reaches new weights in every block.
[Two-rail growth audit](../results/event_lm/trace_read_drive/warm_write_rails_residual_1536.json)

The next comparison starts from the signed write-carry SNN and
full-attention checkpoints at 116,326,400 targets.
Use the original-width controls and the two wider arms at the same
4,915,200 additional targets. The full-attention arm retains its
256-wide backbone and grows its SwiGLU from 2,421 to 5,199 units.
At 25,602,048 parameters, it reproduces the parent logits exactly
on the Metal warm audit. The CPU FP32 audit measures a maximum logit
difference of 0.00001049 after widening. The full runs use seed 9277,
1,200 updates and the same 512-byte training windows. The launcher
scores both corpora at the resulting 121,241,600 cumulative targets.
[Metal full-attention audit](../results/event_lm/trace_read_drive/warm_transformer_swiglu_5199.json) ·
[CPU full-attention audit](../results/event_lm/trace_read_drive/warm_transformer_swiglu_5199_cpu.json)

```bash
audit=checkpoints/work/scale-audits
mkdir -p "$audit"
export PYTORCH_ENABLE_MPS_FALLBACK=0
python experiments/audit_residual_growth.py --enwik8 data/enwik8 --device mps \
  --parent checkpoints/trace-read-drive-study/write_carry_s8237.pt \
  --dim 1536 --json "$audit/warm_snn.json"
python experiments/audit_growth_gradients.py transformer --enwik8 data/enwik8 --device mps \
  --checkpoint checkpoints/trace-read-drive-study/transformer_s8237.pt \
  --seed 9277 --mlp-ratio 30.462890625 --json "$audit/warm_transformer.json"
```

For the CPU Transformer audit use `--device cpu --logit-atol 0.00002`.

```bash
study=checkpoints/trace-read-scale-study
for arm in snn control-snn transformer control-transformer; do
  bash experiments/run_v4_continuation.sh "trace-read-scale-$arm" "$study"
done
```

Use the [read-drive results](TRACE_READ_DRIVE.md) to choose circuit
variants for follow-up. Check native read activity and fixed-weight
interventions alongside loss. Higher headline accuracy can come
from the rest of the network training while a new circuit stays silent.

## Source map

| Location | Contents |
|---|---|
| [Spiking decoder](../snnformer/event_lm.py) and [charge memory](../snnformer/correction_memory.py) | Spike synapses and neurons, routed delays, held reads and the tagged error bank |
| [Full-attention reference](../snnformer/transformer_reference.py) | Matched causal Transformer and feed-forward growth |
| [Trainer](../experiments/train_event_lm.py), [study launcher](../experiments/run_v4_continuation.sh) and [replay](../experiments/replay_event_run.py) | Training, parent initialization and reproducible continuations |
| [Evaluator](../experiments/eval_event_lm.py), [integer pulse audit](../experiments/audit_spike_code.py) and [state interventions](../experiments/probe_event_state.py) | Held-out scores, integer parity, pulse counts and frozen interventions |
| [Growth audit](../experiments/audit_residual_growth.py), [context probe](../experiments/probe_lm_context.py) and [steering probe](../experiments/probe_read_headroom.py) | Warm-start checks, paired histories and spare steering |
| [Generation](../experiments/generate_event_lm.py) and [tests](../tests) | Native CPU decoding and regression checks |
| [Packager](../experiments/package_event_lm.py), [figures](../results/figures) and [machine-readable leaderboard](../results/event_lm/continuation_leaderboard.json) | Archive assembly and generated reports |

Regenerate the release reports from the tracked JSON. Rebuild the
archive from the verified checkpoint paths.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
python experiments/package_event_lm.py --release v8 --release-dir checkpoints/release-v8
```
