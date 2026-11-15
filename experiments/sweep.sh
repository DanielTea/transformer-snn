#!/usr/bin/env bash
# usage: sweep.sh <script.py> <parallel_jobs> <threads_per_job> "<extra args>" variant1 variant2 ...
# Runs each variant as a separate process, at most <parallel_jobs> at a time; logs to results/logs/.
set -u
script=$1; par=$2; thr=$3; extra=$4; shift 4
cd "$(dirname "$0")"
mkdir -p ../results/logs
for v in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 5; done
  log="../results/logs/$(basename "$script" .py)_${v}_$(echo "$extra" | tr -c 'A-Za-z0-9' '_').log"
  OMP_NUM_THREADS=$thr python3 "$script" --variant "$v" --threads "$thr" $extra > "$log" 2>&1 &
done
wait
