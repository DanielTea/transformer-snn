"""Watch an asset's sensors with a stack, as a service would: one capsule per sensor, events when one leaves normal.

The CSV is replayed in order as if its rows arrived live, ``--piece`` rows at a time. Every sensor named by
``--sensors`` is one stream of the stack with its own capsule, written as ``--encoding`` bytes after a quantizer
fitted on the first ``--calibration`` rows, which must be normal operation. The surprise of those rows calibrates a
``SurpriseMonitor``, and from then on every ``watch``, ``alarm`` and ``clear`` is printed and kept in
``<state>/events.jsonl``.

With ``--state`` the capsules, quantizers and monitor are saved there after the last row, and a later call with the
same ``--state`` goes on where this one stopped, with the next rows of a CSV, without fitting anything again. That is
how an asset's capsule outlives one process: the stream is scored exactly as one pass over all of it.

    python experiments/telemetry_monitor.py --csv data/telemetry/skab/valve1/0.csv --delimiter ';' \\
        --time-column datetime --sensors Accelerometer1RMS Current Pressure Temperature \\
        --checkpoint checkpoints/long_streams/l2048-forget/model.pt.averaged --state checkpoints/capsules/pump
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from eval_stream import lengthen_reach
from snnformer.telemetry import ENCODINGS, Quantizer, StreamReader, SurpriseMonitor, encode
from train_event_lm import load_checkpoint

RELEASED_READING = {"reach": 8192, "reach_levels": [4, 5, 6, 7], "teach_levels": list(range(11)),
                    "teach_threshold": 1 << 24}


def read_csv(path, delimiter, sensors, time_column, skip):
    with Path(path).open() as handle:
        rows = list(csv.DictReader(handle, delimiter=delimiter))[skip:]
    missing = [name for name in sensors if rows and name not in rows[0]]
    if missing:
        raise SystemExit(f"{path} has no column {', '.join(missing)}")
    values = np.array([[float(row[name]) for name in sensors] for row in rows])
    times = [row[time_column] if time_column else str(skip + number) for number, row in enumerate(rows)]
    return values, times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--sensors", nargs="+", required=True, help="columns of the CSV, one stream each")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--time-column", help="column printed as the time of an event (default: the row number)")
    parser.add_argument("--skip", type=int, default=0, help="rows of the CSV to leave out at its start")
    parser.add_argument("--calibration", type=int, default=400, help="normal rows that fit levels and the monitor")
    parser.add_argument("--warm", type=int, default=100, help="calibration rows the monitor leaves out")
    parser.add_argument("--encoding", choices=ENCODINGS, default="digits")
    parser.add_argument("--levels", type=int, default=100)
    parser.add_argument("--piece", type=int, default=64, help="rows that arrive at a time")
    parser.add_argument("--span", type=int, default=5, help="span of the monitor's smoothing, in rows")
    parser.add_argument("--hold", type=int, default=3, help="rows a stream must stay past a line to change state")
    parser.add_argument("--no-fast-synapses", action="store_true", help="read without the fast output synapses")
    parser.add_argument("--state", type=Path, help="directory that keeps the capsules, levels and monitor")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    lengthen_reach(model, RELEASED_READING["reach"], RELEASED_READING["reach_levels"])
    if not args.no_fast_synapses:
        model.teach(RELEASED_READING["teach_levels"], RELEASED_READING["teach_threshold"])
    values, times = read_csv(args.csv, args.delimiter, args.sensors, args.time_column, args.skip)
    saved = args.state / "watch.json" if args.state else None
    if saved is not None and saved.exists():
        kept = json.loads(saved.read_text())
        if kept["sensors"] != args.sensors or kept["encoding"] != args.encoding or kept["levels"] != args.levels:
            parser.error(f"{args.state} watches other sensors or another encoding")
        quantizers = [Quantizer(**level) for level in kept["quantizers"]]
        reader = StreamReader.resume(model, args.state / "capsules.pt")
        monitor = SurpriseMonitor.restore(kept["monitor"])
        start = 0
        print(f"resumed {len(args.sensors)} capsules after {reader.samples:,} rows")
    else:
        if len(values) <= args.calibration:
            parser.error("the CSV is not longer than its calibration")
        quantizers = [Quantizer.fit(values[:args.calibration, column], args.levels)
                      for column in range(values.shape[1])]
        reader = StreamReader(model, len(args.sensors))
        monitor = SurpriseMonitor(len(args.sensors), span=args.span, hold=args.hold)
        start = args.calibration
    levels = np.stack([quantizer(values[:, column]) for column, quantizer in enumerate(quantizers)])
    samples = encode(levels, args.encoding, args.levels)
    if start:
        bits = np.concatenate([reader.read(samples[:, at:min(at + args.piece, start)])
                               for at in range(0, start, args.piece)], 1)
        monitor.calibrate(bits, warm=args.warm)
        print(f"calibrated on {start:,} rows: normal surprise "
              + ", ".join(f"{name} {median:.2f}" for name, median in zip(args.sensors, monitor.median))
              + " bits a row")
    log = None
    if args.state:
        args.state.mkdir(parents=True, exist_ok=True)
        log = (args.state / "events.jsonl").open("a")
    offset = reader.samples - start
    for at in range(start, len(values), args.piece):
        bits = reader.read(samples[:, at:at + args.piece])
        for event in monitor.update(bits):
            row = event.sample - offset
            line = {"time": times[row], "sensor": args.sensors[event.stream], "event": event.kind,
                    "score": round(event.score, 2), "value": float(values[row, event.stream]), "row": int(row)}
            print(f"{line['time']}  {line['sensor']:<22} {line['event']:<6} surprise {line['score']:6.2f}  "
                  f"value {line['value']:.4g}", flush=True)
            if log is not None:
                log.write(json.dumps(line) + "\n")
    if log is not None:
        log.close()
        reader.save(args.state / "capsules.pt")
        saved.write_text(json.dumps({"sensors": args.sensors, "encoding": args.encoding, "levels": args.levels,
                                     "quantizers": [{"low": q.low, "high": q.high, "levels": q.levels}
                                                    for q in quantizers],
                                     "monitor": monitor.snapshot(), "rows": reader.samples}, indent=2) + "\n")
        print(f"saved {len(args.sensors)} capsules after {reader.samples:,} rows to {args.state}")


if __name__ == "__main__":
    main()
