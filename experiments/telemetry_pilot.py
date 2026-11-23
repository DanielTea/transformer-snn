"""Read public telemetry through a stack that was trained only on text, and keep its surprise for every sample.

This is the first test of Volley's telemetry claim, before any training on sensor data: does the surprise of a
text-trained stack, reading a sensor as a stream with its fast output synapses learning along it, rise where a
machine leaves its normal? Nothing here takes a gradient. The stack's weights are those of the checkpoint.

Two public datasets with labelled anomalies are read:

* ``nab``: the temperature of an internal component of a large industrial machine from the Numenta Anomaly
  Benchmark (``realKnownCause/machine_temperature_system_failure.csv``), 22,695 samples five minutes apart, with
  NAB's four anomaly windows: a planned shutdown, a fault that is hard to see, and the catastrophic failure it led
  to.
* ``skab``: the Skoltech Anomaly Benchmark, 34 runs of a water-circulation testbed with eight sensors each (two
  vibration RMS, motor current, pressure, two temperatures, voltage, flow) sampled once a second, and the anomaly
  label of every sample.

Every sensor is one stream with its own capsule. A ``Quantizer`` is fitted on the first ``--calibration`` samples of
each run, which are normal by the datasets' own conventions (SKAB's first 400 rows; NAB's samples before its first
window), and every sample is written as ``--encoding`` bytes. The streams of a run are read side by side, in reads
of ``--piece`` samples, and the bits of every sample are kept with the values and labels in ``<out>.npz``;
``<out>.json`` records the settings and the files' hashes. telemetry_evaluate.py scores them against baselines.

Prepare the data under ``data/telemetry`` (see docs/TELEMETRY_PILOT.md) and run, for example::

    python experiments/telemetry_pilot.py --dataset nab --checkpoint checkpoints/long_streams/l2048-forget/model.pt.averaged \\
        --reach 8192 --reach-levels 4 5 6 7 --teach-levels 0 1 2 3 4 5 6 7 8 9 10 --teach-threshold 16777216 \\
        --out results/telemetry/nab_digits_taught
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from eval_stream import lengthen_reach
from snnformer.event_lm import EventLM
from snnformer.telemetry import ENCODINGS, Quantizer, StreamReader, encode, sample_width
from train_event_lm import load_checkpoint

NAB_FILE = "nab/machine_temperature_system_failure.csv"
NAB_KEY = "realKnownCause/machine_temperature_system_failure.csv"
SKAB_GROUPS = ("valve1", "valve2", "other")
SKAB_SENSORS = ("Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple",
                "Voltage", "Volume Flow RateRMS")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_nab(root):
    """The NAB machine temperature as one run: values [samples, 1], the label of every sample, and no change
    points. A sample is anomalous when it lies inside one of NAB's windows for the file."""
    path = root / NAB_FILE
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    times = [datetime.fromisoformat(row["timestamp"]) for row in rows]
    values = np.array([[float(row["value"])] for row in rows])
    windows = json.loads((root / "nab/combined_windows.json").read_text())[NAB_KEY]
    labels = np.zeros(len(rows), dtype=bool)
    starts = np.zeros(len(rows), dtype=bool)
    for start, end in windows:
        start, end = datetime.fromisoformat(start[:19]), datetime.fromisoformat(end[:19])
        inside = np.array([start <= when <= end for when in times])
        labels |= inside
        starts[np.argmax(inside)] = True
    return [{"name": "machine_temperature", "values": values, "labels": labels, "changes": starts,
             "sensors": ["temperature"], "files": [str(path)]}]


def load_skab(root):
    """Every SKAB run: values [samples, 8], the anomaly label and the change-point mark of every sample."""
    runs = []
    for group in SKAB_GROUPS:
        paths = sorted((root / "skab" / group).glob("*.csv"), key=lambda path: int(path.stem))
        for path in paths:
            with path.open() as handle:
                rows = list(csv.DictReader(handle, delimiter=";"))
            values = np.array([[float(row[name]) for name in SKAB_SENSORS] for row in rows])
            labels = np.array([float(row["anomaly"]) > 0 for row in rows])
            changes = np.array([float(row["changepoint"]) > 0 for row in rows])
            runs.append({"name": f"{group}/{path.stem}", "values": values, "labels": labels, "changes": changes,
                         "sensors": list(SKAB_SENSORS), "files": [str(path)]})
    return runs


def read_run(model, run, args):
    """The bits of every sample of every sensor of a run, as [sensors, samples]."""
    values = run["values"]
    calibration = values[:args.calibration]
    quantizers = [Quantizer.fit(calibration[:, sensor], args.levels) for sensor in range(values.shape[1])]
    levels = np.stack([quantizer(values[:, sensor]) for sensor, quantizer in enumerate(quantizers)])
    samples = encode(levels, args.encoding, args.levels)
    reader = StreamReader(model, samples.shape[0])
    if args.reference_piece is None:
        bits = [reader.read(samples[:, start:start + args.piece]) for start in range(0, samples.shape[1], args.piece)]
        return np.concatenate(bits, 1), quantizers
    calibration = args.calibration
    bits = [reader.read(samples[:, start:min(start + args.piece, calibration)])
            for start in range(0, calibration, args.piece)]
    normal = reader.mark()
    for start in range(calibration, samples.shape[1], args.reference_piece):
        reader.rewind(normal)
        bits.append(reader.read(samples[:, start:start + args.reference_piece]))
    return np.concatenate(bits, 1), quantizers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=("nab", "skab"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/telemetry"))
    parser.add_argument("--runs", nargs="+", help="read only these runs of the dataset, by name")
    parser.add_argument("--encoding", choices=ENCODINGS, default="digits")
    parser.add_argument("--levels", type=int, default=100, help="levels of a sample between the ends of normal")
    parser.add_argument("--calibration", type=int, help="samples at the start of every run that fit its levels "
                                                         "(default: 2,000 for nab, 400 for skab)")
    parser.add_argument("--piece", type=int, default=128, help="samples that every stream takes at a time")
    parser.add_argument("--reach", type=int, help="cells that the reads reach back, if more than in training")
    parser.add_argument("--reach-levels", type=int, nargs="+", help="lengthen the reach after these levels only")
    parser.add_argument("--teach-levels", type=int, nargs="+",
                        help="levels whose cells the fast output synapses hear; every arriving byte teaches them")
    parser.add_argument("--teach-threshold", type=int, help="units of fast charge to one spike of an output neuron")
    parser.add_argument("--freeze-after-calibration", action="store_true",
                        help="the fast synapses learn along the calibration samples only and then keep firing with "
                             "what they hold, so that they keep the normal while the rest of the run is read")
    parser.add_argument("--reference-piece", type=int, metavar="SAMPLES",
                        help="score every piece of this many samples after calibration from the capsule that the "
                             "calibration left, not from the state that the stream carried: surprise against normal")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True, help="path without suffix for the .npz and .json")
    args = parser.parse_args()
    if (args.teach_levels is None) != (args.teach_threshold is None):
        parser.error("--teach-levels and --teach-threshold go together")
    if args.reach_levels is not None and args.reach is None:
        parser.error("--reach-levels names the levels that take --reach")
    if args.calibration is None:
        args.calibration = 2000 if args.dataset == "nab" else 400
    sample_width(args.encoding, args.levels)
    for suffix in (".npz", ".json"):
        if args.out.with_suffix(suffix).exists():
            parser.error(f"{args.out.with_suffix(suffix)} already exists")
    torch.set_num_threads(args.threads)

    model, checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if not isinstance(model, EventLM) or not model.cfg.stack:
        parser.error("telemetry is read by a stack of reads and cells")
    if args.reach is not None:
        lengthen_reach(model, args.reach, args.reach_levels)
    if args.teach_levels is not None:
        model.teach(args.teach_levels, args.teach_threshold)
    if args.freeze_after_calibration:
        if args.teach_levels is None:
            parser.error("--freeze-after-calibration freezes the fast synapses of --teach-levels")
        model.freeze_teaching_after(args.calibration * sample_width(args.encoding, args.levels))

    runs = load_nab(args.data_dir) if args.dataset == "nab" else load_skab(args.data_dir)
    if args.runs:
        runs = [run for run in runs if run["name"] in set(args.runs)]
        if not runs:
            parser.error("none of the named runs is in the dataset")
    arrays, records, started = {}, [], time.time()
    for number, run in enumerate(runs):
        began = time.time()
        bits, quantizers = read_run(model, run, args)
        key = f"run{number}"
        arrays[f"{key}_bits"] = bits
        arrays[f"{key}_values"] = run["values"]
        arrays[f"{key}_labels"] = run["labels"]
        arrays[f"{key}_changes"] = run["changes"]
        records.append({"key": key, "name": run["name"], "samples": int(bits.shape[1]), "sensors": run["sensors"],
                        "files": {path: sha256(path) for path in run["files"]},
                        "levels": [{"low": quantizer.low, "high": quantizer.high} for quantizer in quantizers],
                        "seconds": round(time.time() - began, 1)})
        print(f"{run['name']}: {bits.shape[1]:,} samples x {bits.shape[0]} sensors, "
              f"{bits[:, args.calibration:].mean():.3f} bits a sample after calibration, "
              f"{time.time() - began:.0f} s", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".npz"), **arrays)
    record = {"dataset": args.dataset, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
              "run_name": checkpoint.get("run_name"), "encoding": args.encoding, "levels": args.levels,
              "bytes_per_sample": sample_width(args.encoding, args.levels), "calibration_samples": args.calibration,
              "piece_samples": args.piece, "reference_piece_samples": args.reference_piece,
              "reach": {int(level): read.reach for level, read in model.state_reads.items()},
              "taught": None if args.teach_levels is None else {"levels": sorted(args.teach_levels),
                                                                 "threshold": args.teach_threshold,
                                                                 "frozen_after_calibration":
                                                                     args.freeze_after_calibration},
              "threads": args.threads, "seconds": round(time.time() - started, 1), "runs": records}
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
