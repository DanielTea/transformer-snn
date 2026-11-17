#!/usr/bin/env bash
# Replay the coasting-head study behind ShuntFormer v12: fit, train, replay and probe.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash experiments/run_coasting_heads.sh [checkpoint-root]" >&2
  exit 2
fi

root=${1:-checkpoints}
python_exe=${PYTHON:-python}
device=${DEVICE:-cuda}
records=${RECORDS:-$root/coasting-heads-results}
committed=results/event_lm/coasting_heads
v11=checkpoints/surprise-clock-study
setting="orders=1,2,3,4,5,6,7,8;tape=65536"
# The strengths every head run trained beside, and the v11 synapse strengths of the synapse runs.
heads=${HEAD_READOUT:-$committed/readout/heads.json}
synapses=${SYNAPSE_READOUT:-$v11/clock-synapses_s10007.pt}
mixture=data/training-v1/prepared/catalogue.json
mixture_v2=data/training-v2/prepared/catalogue.json
export SNN_BUILD_DIR=${SNN_BUILD_DIR:-$root/build}
mkdir -p "$records" "$SNN_BUILD_DIR"

# The 34.85M decoder of v11: eleven blocks on every byte and four on the surprise clock.
snn=(--variant event_correction_code --no-carry --dim 1408 --hidden 243 --depth 15
  --memory-heads 4 --memory-slots 14 --memory-width 32 --forecast-key-channels 2
  --match-forecast-key --learn-context-keys --spike-resolution 16 --pulse-budget 310000
  --code-loss-weight 0.25 --compact-delays --compact-axon-state --activity-inhibition
  --inhibition-by-expansion --local-trace --trace-write-credit --router-charge-bits 8
  --residual-correction-start 7 --compact-trace-state --delay-group-widths 75,75,47,46
  --trace-arrival selected --trace-arrival-groups 2 --early-readout-weight 0.25
  --clock-start 11 --clock-threshold 35)
transformer=(--variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 42.216796875)

skip() {
  [[ -e $1 ]] && printf 'skip %s (exists)\n' "$1"
}

train() { # CHECKPOINT STEPS LR WARMUP SEED TAG MIXTURE [arguments]
  local checkpoint=$1 steps=$2 lr=$3 warmup=$4 seed=$5 tag=$6 catalogue=$7
  shift 7
  skip "$checkpoint" && return
  mkdir -p "$(dirname "$checkpoint")/results"
  "$python_exe" -u experiments/train_event_lm.py --data data/enwik8 --device "$device" --ctx 512 --bs 8 \
    --validation-only --eval-bs 8 --eval-batches 8 --final-eval-batches 48 --steps "$steps" --warmup "$warmup" \
    --lr "$lr" --seed "$seed" --eval-every $((steps / 30)) --save-every $((steps / 30)) --tag "$tag" \
    --training-mixture "$catalogue" --results-dir "$(dirname "$checkpoint")/results" --checkpoint "$checkpoint" "$@"
  rm -f "$checkpoint.partial"
}

pg19=(--books-dir data/pg19-test --manifest results/event_lm/correction_pg19_full_book_manifest.json
  --calibration-books data/pg19-validation
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json
  --temperatures 0.8 1.0 1.15 1.25 1.4 1.5 1.65 1.8 2.0 2.25)

score() { # CHECKPOINT CONTEXT
  local checkpoint=$1 context=$2 folder name
  folder=$(dirname "$checkpoint")/results
  name=$(basename "$checkpoint" .pt)
  local run=("$python_exe" -u experiments/eval_event_lm.py --device "$device" --context "$context" --history 256
    --batch-size 8 --checkpoint "$checkpoint")
  skip "$folder/enwik8_$name.json" || "${run[@]}" --enwik8 data/enwik8 --json "$folder/enwik8_$name.json"
  skip "$folder/pg19_$name.json" || "${run[@]}" "${pg19[@]}" --json "$folder/pg19_$name.json"
  # Full attention reads 256-byte windows after 256 bytes of history and has no pulses to count.
  [[ $context == 256 ]] && return
  skip "$folder/native_$name.json" || "$python_exe" -u experiments/audit_spike_code.py --device "$device" \
    --split test --enwik8 data/enwik8 --activity-profile --checkpoint "$checkpoint" --json "$folder/native_$name.json"
}

replay() { # LABEL CHECKPOINT [probe arguments]
  local label=$1 checkpoint=$2
  shift 2
  skip "$records/recall_$label.json" && return
  # The decoder's own outputs are recorded once. Synapses and heads are refitted on validation and replayed on test.
  "$python_exe" -u experiments/probe_coasting_heads.py --enwik8 data/enwik8 --checkpoint "$checkpoint" \
    --cache-dir "$root/cache-$label" --work-dir "$SNN_BUILD_DIR" --device "$device" --threads 6 \
    --synapses alone --score-test --setting "$setting" --json "$records/recall_$label.json" "$@"
  rm -rf "$root/cache-$label"
}

after() { # LABEL TRANSFORMER NAME=CHECKPOINT...
  local label=$1 reference=$2 pairs=() checkpoints=()
  shift 2
  for arm in "$@"; do
    pairs+=(--pair "${arm%%=*}" "${arm#*=}" "$reference")
    checkpoints+=("${arm#*=}")
  done
  # Reset windows of 512 bytes: the heads and synapses of a decoder run inside the model here.
  skip "$records/history_gap_$label.json" || "$python_exe" -u experiments/probe_history_gap.py --enwik8 data/enwik8 \
    --split test --device "$device" --windows 2048 "${pairs[@]}" --json "$records/history_gap_$label.json"
  skip "$records/heldout_$label.json" || "$python_exe" -u experiments/eval_byte_transfer.py \
    --prepared data/training-v2/prepared/heldout --device "$device" --batch-size 24 \
    --checkpoint "${checkpoints[@]}" "$reference" --json "$records/heldout_$label.json"
}

# 1. Heads on the frozen v11 decoder. The committed strengths were fitted like this on all of enwik8 validation.
mkdir -p "$records/fit_only"
skip "$records/fit_only/readout_clock.json" || "$python_exe" -u experiments/probe_coasting_heads.py \
  --enwik8 data/enwik8 --checkpoint "$v11/clock_s10007.pt" --cache-dir "$root/cache-v11-clock" \
  --work-dir "$SNN_BUILD_DIR" --device "$device" --setting "$setting" --readout-dir "$records/readout-clock" \
  --json "$records/fit_only/readout_clock.json"
for arm in clock control; do
  skip "$records/fit_only/test_$arm.json" || "$python_exe" -u experiments/probe_coasting_heads.py \
    --enwik8 data/enwik8 --checkpoint "$v11/${arm}_s10007.pt" --cache-dir "$root/cache-v11-$arm" \
    --work-dir "$SNN_BUILD_DIR" --device "$device" --synapses alone joint --score-test --setting "$setting" \
    --json "$records/fit_only/test_$arm.json"
done

# 2. The v11 arms with sixteen times the training: 38,400 updates from the v10 parents.
long=$root/long16-study
for arm in clock depth stride transformer control-transformer; do
  SNN_STEPS=38400 PYTHON=$python_exe DEVICE=$device bash experiments/run_v4_continuation.sh "surprise-clock-$arm" "$long"
  skip "$long/results/synapses_${arm}_s10007.json" || "$python_exe" -u experiments/fit_fast_synapses.py \
    --enwik8 data/enwik8 --cache-dir "$long/cache-$arm" --device "$device" --checkpoint "$long/${arm}_s10007.pt" \
    --groups 4 --cells 1024 --gate 1 --score-test --json "$long/results/synapses_${arm}_s10007.json"
  rm -rf "$long/cache-$arm"
done

# 3. From scratch on the v1 mixture: 90,000 updates, 368,640,000 targets, identical windows.
scratch=$root/scratch-study
train "$scratch/clock_s20001.pt" 90000 0.0003 1000 20001 scratch_clock "$mixture" "${snn[@]}"
train "$scratch/clock-syn_s20001.pt" 90000 0.0003 1000 20001 scratch_clock_syn "$mixture" "${snn[@]}" \
  --synapse-readout "$synapses"
train "$scratch/transformer_s20001.pt" 90000 0.0003 1000 20001 scratch_transformer "$mixture" "${transformer[@]}"
train "$root/v12-scratch/heads_s20001.pt" 90000 0.0003 1000 20001 v12_heads "$mixture" "${snn[@]}" \
  --head-readout "$heads"
# Streaming scores on enwik8 and PG-19 and the native pulse audit. The head decoder carries its tape in the stream.
score "$scratch/clock_s20001.pt" 8192
score "$scratch/clock-syn_s20001.pt" 8192
score "$scratch/transformer_s20001.pt" 256
score "$root/v12-scratch/heads_s20001.pt" 8192

# 4. Three continuations of the 16× clock decoder on identical windows.
parent=$long/clock_s10007.pt
train "$root/v12-long/plain_s30001.pt" 38400 0.00012 96 30001 v12_plain "$mixture" "${snn[@]}" --load-weights "$parent"
train "$root/v12-long/syn_s30001.pt" 38400 0.00012 96 30001 v12_syn "$mixture" "${snn[@]}" --load-weights "$parent" \
  --synapse-readout "$synapses"
train "$root/v12-long/heads_s30001.pt" 38400 0.00012 96 30001 v12_heads "$mixture" "${snn[@]}" --load-weights "$parent" \
  --head-readout "$heads"

# 5. A second pair with another seed and fewer updates.
train "$root/v12-rep/plain_s30003.pt" 18240 0.00012 96 30003 v12_plain "$mixture" "${snn[@]}" --load-weights "$parent"
train "$root/v12-rep/heads_s30003.pt" 18240 0.00012 96 30003 v12_heads "$mixture" "${snn[@]}" --load-weights "$parent" \
  --head-readout "$heads"

# 6. Replays. The head continuation also writes the strengths refitted to its own decoder for round 2.
replay long16-clock "$long/clock_s10007.pt"
replay long16-transformer "$long/transformer_s10007.pt"
replay scratch-clock "$scratch/clock_s20001.pt"
replay scratch-clock-syn "$scratch/clock-syn_s20001.pt"
replay scratch-heads "$root/v12-scratch/heads_s20001.pt" --readout-dir "$records/readout-scratch-heads"
replay scratch-transformer "$scratch/transformer_s20001.pt"
replay long-plain "$root/v12-long/plain_s30001.pt"
replay long-syn "$root/v12-long/syn_s30001.pt"
replay long-heads "$root/v12-long/heads_s30001.pt" --readout-dir "$records/readout-long-heads"
replay rep-plain "$root/v12-rep/plain_s30003.pt"
replay rep-heads "$root/v12-rep/heads_s30003.pt"
# The from-scratch head decoder again with the strengths refitted to it, scored in the stream like the replay.
fitted=$root/v12-scratch/heads-fitted_s20001.pt
skip "$fitted" || "$python_exe" experiments/attach_head_readout.py --checkpoint "$root/v12-scratch/heads_s20001.pt" \
  --readout "${SCRATCH_READOUT:-$committed/readout/scratch-heads/heads_0.json}" --output "$fitted"
score "$fitted" 8192

# 7. Round 2 from the head continuation: the first round's strengths, or strengths refitted to its decoder.
parent=$root/v12-long/heads_s30001.pt
train "$root/v12-round2-same/heads_s30002.pt" 24000 0.00012 96 30002 v12_heads "$mixture" "${snn[@]}" \
  --load-weights "$parent" --head-readout "$heads"
train "$root/v12-round2-refit/heads_s30002.pt" 24000 0.00012 96 30002 v12_heads "$mixture" "${snn[@]}" \
  --load-weights "$parent" --head-readout "${REFIT_READOUT:-$committed/readout/long-heads/heads_0.json}"
replay round2-same "$root/v12-round2-same/heads_s30002.pt"
replay round2-refit "$root/v12-round2-refit/heads_s30002.pt"

# 8. Heads against synapses on the web, mathematics and code mixture. The parent is a from-scratch run on that
#    mixture that was stopped after 30,000 of 90,000 updates. It is restored from the release archive.
parent=$root/scratch-v2-study/clock-syn_s20001.pt.partial
if [[ -f $parent ]]; then
  train "$root/v12-v2pair/syn_s40001.pt" 19200 0.00012 96 40001 v12_syn "$mixture_v2" "${snn[@]}" \
    --load-weights "$parent" --synapse-readout "$synapses"
  train "$root/v12-v2pair/heads_s40001.pt" 19200 0.00012 96 40001 v12_heads "$mixture_v2" "${snn[@]}" \
    --load-weights "$parent" --head-readout "$heads"
  replay v2-syn "$root/v12-v2pair/syn_s40001.pt"
  replay v2-heads "$root/v12-v2pair/heads_s40001.pt"
  after v2pair "$root/scratch-v2-study/transformer_s20001.pt.partial" \
    "syn=$root/v12-v2pair/syn_s40001.pt" "heads=$root/v12-v2pair/heads_s40001.pt"
  skip "$records/heldout_partials.json" || "$python_exe" -u experiments/eval_byte_transfer.py \
    --prepared data/training-v2/prepared/heldout --device "$device" --batch-size 24 \
    --checkpoint "$parent" "$root/v12-snaps/clock-syn_45000.pt" \
    "$root/scratch-v2-study/transformer_s20001.pt.partial" "$scratch/transformer_s20001.pt" \
    --json "$records/heldout_partials.json"
else
  printf 'skip the mixture-v2 pair (%s is not restored)\n' "$parent"
fi

# 9. The gap to full attention by bytes of history, and documents from outside the training mixture.
after scratch "$scratch/transformer_s20001.pt" "clock=$scratch/clock_s20001.pt" \
  "clock-syn=$scratch/clock-syn_s20001.pt" "heads=$root/v12-scratch/heads_s20001.pt"
after long "$long/transformer_s10007.pt" "plain=$root/v12-long/plain_s30001.pt" \
  "syn=$root/v12-long/syn_s30001.pt" "heads=$root/v12-long/heads_s30001.pt"
