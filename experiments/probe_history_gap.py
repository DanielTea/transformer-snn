"""Compare a spiking decoder with full attention by the bytes seen since a stream reset.

Every model scores the same disjoint enwik8 windows from a reset. A pair is a
spiking decoder and its size-matched Transformer. The gap of each pair is
reported by the number of preceding bytes, with intervals that resample
windows. Later pairs are also compared with the first pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from common import _atomic_output, resolve_device
from probe_early_context import paired_interval, score
from train_event_lm import load_checkpoint, training_exposure
from train_lm import ENWIK8_SHA1, load_splits

BINS = ((1, 8), (9, 32), (33, 96), (9, 96), (97, 256), (257, 512), (1, 512))


def _ratio_interval(arm, baseline, resamples):
    ratios = arm.mean(axis=1)[resamples].mean(axis=1) / baseline.mean(axis=1)[resamples].mean(axis=1)
    return np.percentile(ratios, (2.5, 97.5)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--pair", nargs=3, action="append", required=True, metavar=("NAME", "SNN", "TRANSFORMER"),
                        help="a spiking checkpoint and its matched Transformer, the first pair is the baseline")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--windows", type=int, default=1024, help="evenly spaced disjoint reset windows")
    parser.add_argument("--window-bytes", type=int, default=512, help="scored bytes after each reset")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    names = [name for name, _, _ in args.pair]
    if len(set(names)) != len(names):
        parser.error("pair names must be distinct")
    if min(args.windows, args.window_bytes, args.batch_size) < 1:
        parser.error("windows, window size and batch size must be positive")
    if args.json.exists():
        parser.error(f"{args.json} already exists")

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    _, val, test = load_splits(args.enwik8.expanduser())
    data = val if args.split == "val" else test
    span = args.window_bytes + 1
    starts = np.linspace(0, len(data) - span, args.windows, dtype=np.int64)
    if args.windows > 1 and np.diff(starts).min() < span:
        parser.error("the windows must be disjoint")
    windows = torch.from_numpy(data[starts[:, None] + np.arange(span)].astype(np.int64))
    bins = [(low, high) for low, high in BINS if high <= args.window_bytes]
    resamples = np.random.default_rng(5261).integers(args.windows, size=(3000, args.windows))

    losses, models = {}, []
    for path in dict.fromkeys(Path(path) for _, *paths in args.pair for path in paths):
        model, checkpoint = load_checkpoint(path.expanduser(), device)
        losses[path] = score(model, windows, args.batch_size, device)
        with path.expanduser().open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        by_history = {f"{low}-{high}": float(losses[path][:, low - 1:high].mean()) for low, high in bins}
        models.append({"checkpoint": str(path), "sha256": digest, "run_name": checkpoint["run_name"],
                       "parameters": sum(parameter.numel() for parameter in model.parameters()),
                       "training_targets": training_exposure(checkpoint), "bpb_by_preceding_bytes": by_history})
        print(f"{path.name} | " + " | ".join(f"{label} {bpb:.4f}" for label, bpb in by_history.items()), flush=True)
        del model

    gaps = {name: losses[Path(snn)] - losses[Path(reference)] for name, snn, reference in args.pair}
    baseline = gaps[names[0]]
    pairs = []
    for name, snn, reference in args.pair:
        rows = []
        for low, high in bins:
            gap, first = gaps[name][:, low - 1:high], baseline[:, low - 1:high]
            row = {"preceding_bytes": [low, high], "targets": gap.size, "gap_bpb": float(gap.mean()),
                   "paired_95_bpb": paired_interval(gap, resamples)}
            if name != names[0]:
                row.update(gap_change_bpb=float((gap - first).mean()),
                           gap_change_paired_95_bpb=paired_interval(gap - first, resamples),
                           gap_ratio=float(gap.mean() / first.mean()),
                           gap_ratio_95=_ratio_interval(gap, first, resamples))
            rows.append(row)
        pairs.append({"name": name, "snn": snn, "transformer": reference, "baseline": names[0], "histories": rows})
        print(f"{name} | " + " | ".join(f"{row['preceding_bytes'][0]}-{row['preceding_bytes'][1]} "
                                        f"{row['gap_bpb']:+.4f}" for row in rows), flush=True)

    result = {"data_sha1": ENWIK8_SHA1, "split": args.split, "device": str(device), "windows": args.windows,
              "window_bytes": args.window_bytes, "window_starts": starts.tolist(), "resamples": len(resamples),
              "models": models, "pairs": pairs}
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
