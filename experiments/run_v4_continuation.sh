#!/usr/bin/env bash
# Train and score a continuation from the held-charge studies.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash experiments/run_v4_continuation.sh STUDY [output-dir]" >&2
  echo "Studies: trace-coincidence trace-write-confirm extend-snn extend-transformer" >&2
  echo "         expansion-snn expansion-transformer expansion-control expansion-steering" >&2
  echo "         scale-data-snn scale-data-transformer" >&2
  exit 2
fi

study=$1
output_dir=${2:-checkpoints/work/$study}
python_exe=${PYTHON:-python}
device=${DEVICE:-mps}
export PYTORCH_ENABLE_MPS_FALLBACK=0

training=(--data data/enwik8 --device "$device" --ctx 512 --bs 8
  --validation-only --eval-bs 8 --eval-batches 8 --final-eval-batches 48)
spiking=(--variant event_correction_code --no-carry --dim 768 --hidden 187 --depth 7
  --memory-heads 4 --memory-slots 14 --memory-width 32 --forecast-key-channels 2
  --match-forecast-key --learn-context-keys --spike-resolution 16 --pulse-budget 310000
  --code-loss-weight 0.25 --compact-delays --compact-axon-state --activity-inhibition
  --inhibition-by-expansion --local-trace --trace-write-credit)

case "$study" in
  trace-coincidence)
    training+=("${spiking[@]}" --grow-residual --trace-coincidence-start 384
      --initialize-from checkpoints/scale-3260k-study/trace_s0.pt
      --steps 1600 --warmup 64 --lr 0.0004 --seed 711 --eval-every 200
      --tag scale_6510k_trace_coincidence)
    name=coincidence_s711
    ;;
  extend-snn)
    training+=("${spiking[@]}" --initialize-from checkpoints/grown-6510k-residual-study/grown_s711.pt)
    name=trace_s1109
    ;;
  extend-transformer)
    training+=(--variant transformer --dim 256 --depth 4 --heads 4 --mlp-ratio 10.2890625
      --initialize-from checkpoints/grown-6510k-residual-study/transformer_s711.pt)
    name=transformer_s1109
    ;;
  scale-data-snn)
    training+=("${spiking[@]}" --initialize-from checkpoints/scale-6510k-data-study/trace_s1109.pt)
    name=trace_s2503
    ;;
  scale-data-transformer)
    training+=(--variant transformer --dim 256 --depth 4 --heads 4 --mlp-ratio 10.2890625
      --initialize-from checkpoints/scale-6510k-data-study/transformer_s1109.pt)
    name=transformer_s2503
    ;;
  expansion-snn|expansion-steering)
    training+=("${spiking[@]}" --hidden 256 --grow-expansion
      --initialize-from checkpoints/scale-6510k-data-study/trace_s1109.pt)
    name=expanded_s1801
    if [[ $study == expansion-steering ]]; then
      training+=(--steering-gradient-at-zero)
      name=expanded_steering_s1801
    fi
    ;;
  expansion-transformer)
    training+=(--variant transformer --dim 256 --depth 4 --heads 4 --mlp-ratio 14.537109375
      --grow-feed-forward --initialize-from checkpoints/scale-6510k-data-study/transformer_s1109.pt)
    name=transformer_s1801
    ;;
  expansion-control)
    training+=("${spiking[@]}" --initialize-from checkpoints/scale-6510k-data-study/trace_s1109.pt)
    name=control_s1801
    ;;
  trace-write-confirm)
    training+=("${spiking[@]}" --trace-write-confirm
      --initialize-from checkpoints/scale-6510k-data-study/trace_s1109.pt)
    name=confirm_s1801
    ;;
  *)
    echo "Unknown continuation: $study" >&2
    exit 2
    ;;
esac

if [[ $study == extend-* ]]; then
  training+=(--steps 4000 --warmup 128 --lr 0.0003 --seed 1109 --eval-every 400
    --tag scale_6510k_49152k_targets)
elif [[ $study == scale-data-* ]]; then
  training+=(--steps 4000 --warmup 128 --lr 0.0003 --seed 2503 --eval-every 400
    --tag scale_6510k_65536k_targets)
elif [[ $study == expansion-* || $study == trace-write-confirm ]]; then
  tag=scale_6510k_trace_write_confirm
  if [[ $study == expansion-* ]]; then
    tag="scale_8740k_expansion_${study#expansion-}"
  fi
  training+=(--steps 1600 --warmup 64 --lr 0.0003 --seed 1801 --eval-every 200
    --tag "$tag")
fi

checkpoint="$output_dir/$name.pt"
records="$output_dir/results"
mkdir -p "$records"
"$python_exe" -u experiments/train_event_lm.py "${training[@]}" \
  --results-dir "$records" --checkpoint "$checkpoint"

context=8192
if [[ $study == *-transformer ]]; then
  context=256
fi
score=("$python_exe" -u experiments/eval_event_lm.py --device "$device"
  --context "$context" --history 256 --batch-size 8 --checkpoint "$checkpoint")
"${score[@]}" --enwik8 data/enwik8 --json "$records/enwik8_$name.json"
"${score[@]}" --books-dir data/pg19-test \
  --manifest results/event_lm/correction_pg19_full_book_manifest.json \
  --calibration-books data/pg19-validation \
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json \
  --temperatures 0.8 1.0 1.15 1.25 1.4 1.5 1.65 1.8 2.0 2.25 \
  --json "$records/pg19_$name.json"

if [[ $study != *-transformer ]]; then
  "$python_exe" -u experiments/audit_spike_code.py --device "$device" --split test \
    --enwik8 data/enwik8 --activity-profile --checkpoint "$checkpoint" --json "$records/native_$name.json"
fi
if [[ $study == trace-coincidence || $study == trace-write-confirm || $study == expansion-snn || $study == expansion-steering ]]; then
  "$python_exe" -u experiments/probe_event_state.py --device "$device" --enwik8 data/enwik8 \
    --chunk-size 512 --targets 65536 --checkpoint "$checkpoint" --json "$records/probe_$name.json"
fi
if [[ $study == trace-write-confirm ]]; then
  "$python_exe" -u experiments/profile_charge_trace.py --split validation --bytes 16384 \
    --enwik8 data/enwik8 --checkpoint "$checkpoint" --json "$records/trace_ages_$name.json"
fi
