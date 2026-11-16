#!/usr/bin/env bash
# Run a continuation from the v4 handoff and score the resulting checkpoint.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash experiments/run_v4_continuation.sh {trace-coincidence|extend-snn|extend-transformer} [output-dir]" >&2
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
  *)
    echo "Unknown continuation: $study" >&2
    exit 2
    ;;
esac

if [[ $study == extend-* ]]; then
  training+=(--steps 4000 --warmup 128 --lr 0.0003 --seed 1109 --eval-every 400
    --tag scale_6510k_49152k_targets)
fi

checkpoint="$output_dir/$name.pt"
records="$output_dir/results"
mkdir -p "$records"
"$python_exe" -u experiments/train_event_lm.py "${training[@]}" \
  --results-dir "$records" --checkpoint "$checkpoint"

context=8192
if [[ $study == extend-transformer ]]; then
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

if [[ $study != extend-transformer ]]; then
  "$python_exe" -u experiments/audit_spike_code.py --device "$device" --split test \
    --enwik8 data/enwik8 --activity-profile --checkpoint "$checkpoint" --json "$records/native_$name.json"
fi
if [[ $study == trace-coincidence ]]; then
  "$python_exe" -u experiments/probe_event_state.py --device "$device" --enwik8 data/enwik8 \
    --chunk-size 512 --targets 65536 --checkpoint "$checkpoint" --json "$records/probe_$name.json"
fi
