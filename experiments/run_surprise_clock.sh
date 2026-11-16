#!/usr/bin/env bash
# Replay the surprise study behind ShuntFormer v11: measure, train, attach, score and probe.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash experiments/run_surprise_clock.sh [output-dir]" >&2
  exit 2
fi

out=${1:-checkpoints/surprise-clock-study}
python_exe=${PYTHON:-python}
device=${DEVICE:-mps}
parents=checkpoints/residual-scale-study
records="$out/results"
cache="$out/cache"
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p "$records"

run_report() {
  local destination=$1
  shift
  if [[ -f $destination ]]; then
    printf 'skip %s (exists)\n' "$destination"
  else
    "$@" --json "$destination"
  fi
}

# 1. Where the frozen v10 decoder is surprised, and what an exact successor memory could add.
run_report "$records/surprise_validation.json" "$python_exe" -u experiments/probe_surprise.py \
  --enwik8 data/enwik8 --split val --device "$device" --cache-dir "$cache" \
  --checkpoint "$parents/wide-snn_s9631.pt" "$parents/wide-transformer_s9631.pt"

# 2. Six matched continuations on identical windows, each scored on enwik8 and PG-19.
for arm in control depth clock stride transformer control-transformer; do
  PYTHON=$python_exe DEVICE=$device bash experiments/run_v4_continuation.sh "surprise-clock-$arm" "$out"
done

# 3. Fast synapses on the frozen decoders. The sweep fits on the first half of validation and scores the second.
synapses=(--enwik8 data/enwik8 --cache-dir "$cache" --device "$device")
run_report "$records/synapse_sweep.json" "$python_exe" -u experiments/fit_fast_synapses.py "${synapses[@]}" \
  --checkpoint "$out/clock_s10007.pt" --groups 4 --cells 256 1024 4096 16384 --gate 0 1
run_report "$records/synapse_gate_sweep.json" "$python_exe" -u experiments/fit_fast_synapses.py "${synapses[@]}" \
  --checkpoint "$out/clock_s10007.pt" --groups 4 --cells 1024 --gate 2 4 8
run_report "$records/synapse_group_sweep.json" "$python_exe" -u experiments/fit_fast_synapses.py "${synapses[@]}" \
  --checkpoint "$out/clock_s10007.pt" --groups 2 8 --cells 512 2048 --gate 1
for arm in control clock; do
  name=$(basename "$records"/*_surprise_clock_"$arm".json .json)
  run_report "$records/synapses_${arm}_s10007.json" "$python_exe" -u experiments/fit_fast_synapses.py \
    "${synapses[@]}" --checkpoint "$out/${arm}_s10007.pt" --groups 4 --cells 1024 --gate 1 --score-test \
    --output "$out/${arm}-synapses_s10007.pt" --training-record "$records/$name.json"
done
# The same fit on the other grown decoders and on the Transformers' outputs. These write no checkpoint.
for arm in depth stride transformer control-transformer; do
  run_report "$records/synapses_${arm}_s10007.json" "$python_exe" -u experiments/fit_fast_synapses.py \
    "${synapses[@]}" --checkpoint "$out/${arm}_s10007.pt" --groups 4 --cells 1024 --gate 1 --score-test
done

# 4. Score the decoders that now carry synapses the same way as the trained arms.
for arm in control-synapses clock-synapses; do
  checkpoint="$out/${arm}_s10007.pt"
  score=("$python_exe" -u experiments/eval_event_lm.py --device "$device" --context 8192 --history 256
    --batch-size 8 --checkpoint "$checkpoint")
  run_report "$records/enwik8_${arm}_s10007.json" "${score[@]}" --enwik8 data/enwik8
  run_report "$records/pg19_${arm}_s10007.json" "${score[@]}" --books-dir data/pg19-test \
    --manifest results/event_lm/correction_pg19_full_book_manifest.json \
    --calibration-books data/pg19-validation \
    --calibration-manifest results/event_lm/correction_pg19_validation_manifest.json \
    --temperatures 0.8 1.0 1.15 1.25 1.4 1.5 1.65 1.8 2.0 2.25
  run_report "$records/native_${arm}_s10007.json" "$python_exe" -u experiments/audit_spike_code.py \
    --device "$device" --split test --enwik8 data/enwik8 --activity-profile --checkpoint "$checkpoint"
done

# 5. Fixed-weight probes: mute or displace the clock, mute the synapses, and vary the history.
run_report "$records/clock_probe_test.json" "$python_exe" -u experiments/probe_clock.py --enwik8 data/enwik8 \
  --split test --bytes 1048576 --device "$device" \
  --checkpoint "$out"/{clock,stride,control-synapses,clock-synapses}_s10007.pt
# Paired reset windows: the gap of each spiking arm to its size-matched Transformer by bytes of history.
pairs=()
for arm in control depth stride clock control-synapses clock-synapses; do
  reference=transformer
  if [[ $arm == control* ]]; then
    reference=control-transformer
  fi
  pairs+=(--pair "$arm" "$out/${arm}_s10007.pt" "$out/${reference}_s10007.pt")
done
run_report "$records/history_gap_test.json" "$python_exe" -u experiments/probe_history_gap.py --enwik8 data/enwik8 \
  --split test --device "$device" --windows 2048 "${pairs[@]}"
# The same target bytes with more and more history, past the 512 bytes that full attention reads.
run_report "$records/long_history_test.json" "$python_exe" -u experiments/probe_lm_context.py --event-lm \
  --data data/enwik8 --split test --device "$device" --batch-size 4 --histories 128 512 2048 8192 \
  --checkpoint "$out"/{control,control-synapses,clock,clock-synapses}_s10007.pt
run_report "$records/byte_transfer.json" "$python_exe" -u experiments/eval_byte_transfer.py --device "$device" \
  --batch-size 24 --prepared data/transfer-v1/prepared \
  --checkpoint "$out"/{transformer,control-transformer,control,depth,stride,clock,control-synapses,clock-synapses}_s10007.pt
