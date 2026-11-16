#!/usr/bin/env bash
# Train and score a continuation from the held-charge studies.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash experiments/run_v4_continuation.sh STUDY [output-dir]" >&2
  echo "Studies: trace-coincidence trace-write-confirm extend-snn extend-transformer" >&2
  echo "         expansion-snn expansion-transformer expansion-control expansion-steering" >&2
  echo "         scale-data-snn scale-data-transformer" >&2
  echo "         residual-growth-snn residual-growth-transformer residual-growth-control" >&2
  echo "         residual-growth-control-transformer" >&2
  echo "         redistribute-inhibition teacher-guidance late-steering finer-router-charge" >&2
  echo "         tag-read-competition tag-read-active-start tag-read-supervision" >&2
  echo "         depth-growth-snn depth-growth-control depth-growth-wide depth-growth-transformer" >&2
  echo "         depth-growth-corrections depth-growth-control-transformer" >&2
  echo "         trace-transport trace-group-thresholds trace-transport-control trace-transport-transformer" >&2
  echo "         trace-fine-group-thresholds" >&2
  echo "         trace-tap-control trace-tap-fixed trace-tap-race trace-tap-transformer" >&2
  echo "         packed-width-snn packed-width-control packed-width-transformer packed-width-control-transformer" >&2
  echo "         packed-width-present-transfer packed-width-delayed-transfer packed-width-short-groups" >&2
  echo "         trace-read-drive-control trace-read-drive-write trace-read-drive-present trace-read-drive-transformer" >&2
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
  residual-growth-snn)
    training+=("${spiking[@]}" --dim 1536 --grow-residual
      --initialize-from checkpoints/scale-6510k-next-data-study/trace_s2503.pt)
    name=residual_s3307
    ;;
  residual-growth-transformer)
    training+=(--variant transformer --dim 256 --depth 4 --heads 4 --mlp-ratio 22.716796875
      --grow-feed-forward --initialize-from checkpoints/scale-6510k-next-data-study/transformer_s2503.pt)
    name=transformer_s3307
    ;;
  residual-growth-control)
    training+=("${spiking[@]}" --initialize-from checkpoints/scale-6510k-next-data-study/trace_s2503.pt)
    name=control_s3307
    ;;
  residual-growth-control-transformer)
    training+=(--variant transformer --dim 256 --depth 4 --heads 4 --mlp-ratio 10.2890625
      --initialize-from checkpoints/scale-6510k-next-data-study/transformer_s2503.pt)
    name=control_transformer_s3307
    ;;
  tag-read-competition|tag-read-active-start|tag-read-supervision)
    training+=("${spiking[@]}" --tag-read-competition
      --initialize-from checkpoints/scale-6510k-next-data-study/trace_s2503.pt)
    case "$study" in
      tag-read-competition) name=tag_reads_s3307 ;;
      tag-read-active-start)
        training+=(--tag-read-initial-gain 0.25)
        name=tag_active_s3307
        ;;
      tag-read-supervision)
        training+=(--tag-read-initial-gain 0.25 --bank-read-rank-weight 0.10)
        name=tag_supervised_s3307
        ;;
    esac
    ;;
  late-steering)
    training+=("${spiking[@]}" --late-steering
      --initialize-from checkpoints/scale-6510k-next-data-study/trace_s2503.pt)
    name=late_s3307
    ;;
  finer-router-charge)
    training+=("${spiking[@]}" --router-charge-bits 8 --preserve-router-charge
      --initialize-from checkpoints/scale-6510k-next-data-study/trace_s2503.pt)
    name=fine_charge_s3307
    ;;
  depth-growth-snn|depth-growth-control|depth-growth-wide|depth-growth-corrections)
    training+=("${spiking[@]}" --router-charge-bits 8
      --initialize-from checkpoints/finer-router-charge-study/fine_charge_s3307.pt)
    case "$study" in
      depth-growth-snn)
        training+=(--depth 11 --grow-depth --residual-correction-start 7)
        name=depth_s4603
        ;;
      depth-growth-control) name=control_s4603 ;;
      depth-growth-wide)
        training+=(--dim 1174 --grow-residual)
        name=wide_s4603
        ;;
      depth-growth-corrections)
        training+=(--residual-correction-start 0)
        name=corrections_s4603
        ;;
    esac
    ;;
  depth-growth-transformer)
    training+=(--variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 10.576171875
      --grow-depth --initialize-from checkpoints/scale-6510k-short-control-study/control_transformer_s3307.pt)
    name=transformer_s4603
    ;;
  depth-growth-control-transformer)
    training+=(--variant transformer --dim 256 --depth 4 --heads 4 --mlp-ratio 10.2890625
      --initialize-from checkpoints/scale-6510k-short-control-study/control_transformer_s3307.pt)
    name=control_transformer_s4603
    ;;
  trace-transport|trace-group-thresholds|trace-fine-group-thresholds|trace-transport-control)
    training+=("${spiking[@]}" --depth 11 --router-charge-bits 8 --residual-correction-start 7
      --initialize-from checkpoints/depth-growth-study/depth_s4603.pt)
    case "$study" in
      trace-transport)
        training+=(--trace-integration)
        name=transport_s5801
        ;;
      trace-group-thresholds)
        training+=(--learn-inhibition-thresholds)
        name=thresholds_s5801
        ;;
      trace-fine-group-thresholds)
        training+=(--learn-inhibition-thresholds --inhibition-threshold-bits 8)
        name=fine_thresholds_s5801
        ;;
      trace-transport-control) name=control_s5801 ;;
    esac
    ;;
  trace-transport-transformer)
    training+=(--variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 10.576171875
      --initialize-from checkpoints/depth-growth-study/transformer_s4603.pt)
    name=transformer_s5801
    ;;
  trace-tap-control|trace-tap-fixed|trace-tap-race)
    training+=("${spiking[@]}" --depth 11 --router-charge-bits 8 --residual-correction-start 7
      --initialize-from checkpoints/trace-transport-study/control_s5801.pt)
    arm=${study#trace-tap-}
    if [[ $arm != control ]]; then
      training+=(--trace-tap-read "$arm")
    fi
    name="${arm}_s6899"
    ;;
  trace-tap-transformer)
    training+=(--variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 10.576171875
      --initialize-from checkpoints/trace-transport-study/transformer_s5801.pt)
    name=transformer_s6899
    ;;
  packed-width-snn|packed-width-control|packed-width-present-transfer|packed-width-delayed-transfer|packed-width-short-groups)
    training+=("${spiking[@]}" --depth 11 --router-charge-bits 8 --residual-correction-start 7
      --initialize-from checkpoints/trace-tap-study/control_s6899.pt)
    if [[ $study != packed-width-control ]]; then
      training+=(--hidden 243 --grow-expansion --compact-trace-state)
    fi
    case "$study" in
      packed-width-snn) name=wide_s7907 ;;
      packed-width-control) name=control_s7907 ;;
      packed-width-short-groups)
        training+=(--delay-group-widths 75,75,47,46)
        name=short_groups_s7907
        ;;
      packed-width-*-transfer)
        source=${study#packed-width-}
        source=${source%-transfer}
        training+=(--steering-transfer "$source")
        name="${source}_transfer_s7907"
        ;;
    esac
    ;;
  packed-width-transformer|packed-width-control-transformer)
    training+=(--variant transformer --dim 256 --depth 6 --heads 4
      --initialize-from checkpoints/trace-tap-study/transformer_s6899.pt)
    if [[ $study == packed-width-transformer ]]; then
      training+=(--mlp-ratio 14.185546875 --grow-feed-forward)
      name=transformer_s7907
    else
      training+=(--mlp-ratio 10.576171875)
      name=control_transformer_s7907
    fi
    ;;
  trace-read-drive-control|trace-read-drive-write|trace-read-drive-present)
    training+=("${spiking[@]}" --depth 11 --hidden 243 --router-charge-bits 8 --residual-correction-start 7
      --compact-trace-state --delay-group-widths 75,75,47,46
      --initialize-from checkpoints/packed-width-study/short_groups_s7907.pt)
    arm=${study#trace-read-drive-}
    if [[ $arm != control ]]; then
      training+=(--trace-read-drive "$arm")
    fi
    name="${arm}_s8237"
    ;;
  trace-read-drive-transformer)
    training+=(--variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 14.185546875
      --initialize-from checkpoints/packed-width-study/transformer_s7907.pt)
    name=transformer_s8237
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
  redistribute-inhibition)
    training+=("${spiking[@]}" --redistribute-inhibition
      --initialize-from checkpoints/scale-6510k-data-study/trace_s1109.pt)
    name=inhibition_s1801
    ;;
  teacher-guidance)
    training+=("${spiking[@]}" --initialize-from checkpoints/scale-6510k-data-study/trace_s1109.pt
      --teacher-checkpoint checkpoints/scale-6510k-data-study/transformer_s1109.pt)
    name=guided_s1801
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
elif [[ $study == depth-growth-* ]]; then
  training+=(--steps 2400 --warmup 96 --lr 0.0002 --seed 4603 --eval-every 300
    --tag "${study//-/_}")
elif [[ $study == trace-transport* || $study == trace-group-thresholds || $study == trace-fine-group-thresholds ]]; then
  training+=(--steps 2400 --warmup 96 --lr 0.0002 --seed 5801 --eval-every 300
    --tag "trace_transport_${name%_s5801}")
elif [[ $study == trace-tap-* ]]; then
  training+=(--steps 2400 --warmup 96 --lr 0.0002 --seed 6899 --eval-every 300
    --tag "trace_tap_${name%_s6899}")
elif [[ $study == packed-width-* ]]; then
  training+=(--steps 2400 --warmup 96 --lr 0.0002 --seed 7907 --eval-every 300
    --tag "${study//-/_}")
elif [[ $study == trace-read-drive-* ]]; then
  training+=(--steps 1200 --warmup 48 --lr 0.00008 --seed 8237 --eval-every 150
    --tag "${study//-/_}")
elif [[ $study == residual-growth-control-transformer ]]; then
  training+=(--steps 1600 --warmup 64 --lr 0.0003 --seed 3307 --eval-every 200
    --tag scale_6510k_residual_transformer_control)
elif [[ $study == residual-growth-* ]]; then
  training+=(--steps 1600 --warmup 64 --lr 0.0003 --seed 3307 --eval-every 200
    --tag "scale_13030k_residual_${study#residual-growth-}")
elif [[ $study == tag-read-* || $study == late-steering || $study == finer-router-charge ]]; then
  training+=(--steps 1600 --warmup 64 --lr 0.0003 --seed 3307 --eval-every 200
    --tag "${study//-/_}")
elif [[ $study == expansion-* || $study == trace-write-confirm || $study == redistribute-inhibition ||
        $study == teacher-guidance ]]; then
  case "$study" in
    expansion-*) tag="scale_8740k_expansion_${study#expansion-}" ;;
    trace-write-confirm) tag=scale_6510k_trace_write_confirm ;;
    redistribute-inhibition) tag=scale_6510k_redistribute_inhibition ;;
    teacher-guidance) tag=scale_6510k_teacher_guidance ;;
  esac
  training+=(--steps 1600 --warmup 64 --lr 0.0003 --seed 1801 --eval-every 200
    --tag "$tag")
fi

checkpoint="$output_dir/$name.pt"
records="$output_dir/results"
mkdir -p "$records"
"$python_exe" -u experiments/train_event_lm.py "${training[@]}" \
  --results-dir "$records" --checkpoint "$checkpoint"
if [[ ! -f $checkpoint ]]; then
  printf 'Missing checkpoint %s. Restore it or use a fresh output directory.\n' "$checkpoint" >&2
  exit 1
fi

run_report() {
  local destination=$1
  shift
  if [[ -f $destination ]]; then
    printf 'skip %s (exists)\n' "$destination"
  else
    "$@" --json "$destination"
  fi
}

context=8192
if [[ $study == *-transformer ]]; then
  context=256
fi
score=("$python_exe" -u experiments/eval_event_lm.py --device "$device"
  --context "$context" --history 256 --batch-size 8 --checkpoint "$checkpoint")
run_report "$records/enwik8_$name.json" "${score[@]}" --enwik8 data/enwik8
run_report "$records/pg19_$name.json" "${score[@]}" --books-dir data/pg19-test \
  --manifest results/event_lm/correction_pg19_full_book_manifest.json \
  --calibration-books data/pg19-validation \
  --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json \
  --temperatures 0.8 1.0 1.15 1.25 1.4 1.5 1.65 1.8 2.0 2.25

if [[ $study != *-transformer ]]; then
  run_report "$records/native_$name.json" "$python_exe" -u experiments/audit_spike_code.py \
    --device "$device" --split test --enwik8 data/enwik8 --activity-profile --checkpoint "$checkpoint"
fi
if [[ $study == trace-coincidence || $study == trace-write-confirm || $study == redistribute-inhibition ||
      $study == tag-read-* || $study == late-steering || $study == finer-router-charge ||
      $study == expansion-snn || $study == expansion-steering || $study == residual-growth-snn ||
      $study == depth-growth-snn || $study == depth-growth-wide || $study == trace-transport ||
      $study == trace-group-thresholds || $study == trace-fine-group-thresholds || $study == trace-transport-control ||
      $study == trace-tap-fixed || $study == trace-tap-race ||
      ($study == trace-read-drive-* && $study != *transformer) ||
      ($study == packed-width-* && $study != *transformer) ]]; then
  run_report "$records/probe_$name.json" "$python_exe" -u experiments/probe_event_state.py \
    --device "$device" --enwik8 data/enwik8 --chunk-size 512 --targets 65536 --checkpoint "$checkpoint"
fi
if [[ $study == trace-write-confirm ]]; then
  run_report "$records/trace_ages_$name.json" "$python_exe" -u experiments/profile_charge_trace.py \
    --split validation --bytes 16384 --enwik8 data/enwik8 --checkpoint "$checkpoint"
fi
