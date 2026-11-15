"""Measure time since the last cell write at learned probes on enwik8."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from probe_lm_context import load_model
from snnformer.pulse_memory import PulseMemory, _THRESHOLD
from train_lm import DATA, ENWIK8_SHA1, load_splits


@torch.inference_mode()
def inspect(model, windows, batch_size, device, min_age):
    banks = [module for module in model.modules() if isinstance(module, PulseMemory)]
    if len(banks) != 1 or not banks[0].carry_reads:
        raise ValueError("expected a single nondestructive paired-pulse memory bank")
    bank = banks[0]
    width = windows.shape[1]
    positions = torch.arange(1, width + 1, device=device)[None, :, None]
    ages = torch.zeros(width, dtype=torch.long)
    dependent_ages = torch.zeros_like(ages)
    totals = dict(probe_events=0, write_events=0, read_spikes=0)
    capture = []
    original_events = bank._events

    def record_events(*args, **kwargs):
        events = original_events(*args, **kwargs)
        capture.append(events)
        return events

    with patch.object(bank, "_events", record_events):
        for batch in windows.split(batch_size):
            capture.clear()
            stats = {}
            model(batch.to(device), memory_stats=stats)
            (probe_gate, pulse, proposals, charge), = capture
            probe = probe_gate.bool()
            write = bank._accepted_writes(proposals).bool()

            latest = torch.where(write, positions, 0).cummax(dim=1).values
            prior = F.pad(latest[:, :-1], (0, 0, 1, 0))
            phase = F.pad(charge, (0, 0, 1, 0)).gather(1, prior)
            age = positions - prior
            initialized = probe & (prior != 0)
            # Hold current pulse and routing fixed, then ask if setting the
            # phase to zero would suppress a binary bit or change its timing bin.
            if bank.timing is None:
                emitted = probe & (phase + pulse >= _THRESHOLD)
                dependent = emitted & (pulse < _THRESHOLD)
            else:
                events = bank.timing(phase, pulse, probe_gate)
                empty_events = bank.timing(torch.zeros_like(phase), pulse, probe_gate)
                emitted = events.bool().any(-1)
                dependent = (events != empty_events).any(-1)
            ages += torch.bincount(age[initialized].cpu(), minlength=width)
            dependent_ages += torch.bincount(age[dependent].cpu(), minlength=width)

            counted = dict(probe_events=int(probe.sum()), write_events=int(write.sum()),
                           read_spikes=int(emitted.sum()))
            if any(count != int(stats[name]) for name, count in counted.items()):
                raise RuntimeError("cell-age reconstruction disagrees with model event counts")
            for name, count in counted.items():
                totals[name] += count

    ages = ages.numpy()
    initialized = int(ages.sum())
    probes = totals["probe_events"]
    uninitialized = probes - initialized
    median, oldest = None, None
    if initialized:
        ranks = [(initialized + 1) // 2, (initialized + 2) // 2]
        median = float(np.searchsorted(ages.cumsum(), ranks).mean())
        oldest = int(np.flatnonzero(ages)[-1])
    old = int(ages[min_age:].sum())
    old_dependent = int(dependent_ages[min_age:].sum())
    return {
        "memory_cells": bank.cells,
        **totals,
        "initialized_probes": initialized,
        "uninitialized_probes": uninitialized,
        "uninitialized_probe_fraction": uninitialized / probes if probes else None,
        "median_initialized_age_bytes": median,
        "max_initialized_age_bytes": oldest,
        "minimum_old_age_bytes": min_age,
        "old_probes": old,
        "old_probe_fraction_of_all_probes": old / probes if probes else None,
        "old_probe_fraction_of_initialized_probes": old / initialized if initialized else None,
        "phase_dependent_spikes": int(dependent_ages.sum()),
        "old_phase_dependent_spikes": old_dependent,
        "old_phase_dependent_spike_fraction_of_all_probes": old_dependent / probes if probes else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", DATA)))
    parser.add_argument("--window-count", type=int, default=128)
    parser.add_argument("--window-bytes", type=int, default=1024)
    parser.add_argument("--min-age", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing JSON result")
    args = parser.parse_args()
    if min(args.window_count, args.window_bytes, args.min_age, args.batch_size, args.threads) < 1:
        parser.error("sizes, age and thread count must be positive")
    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        model, run = load_model(args.checkpoint, device)
        _, _, test = load_splits(args.data.expanduser())
        if args.window_bytes * args.window_count > len(test):
            raise ValueError("windows would overlap or exceed the test split")
        starts = np.linspace(0, len(test) - args.window_bytes, args.window_count, dtype=np.int64)
        windows = torch.from_numpy(test[starts[:, None] + np.arange(args.window_bytes)].astype(np.int64))
        record = {
            "checkpoint": str(args.checkpoint),
            "run_name": run,
            "data_sha1": ENWIK8_SHA1,
            "split": "test",
            "split_bytes": len(test),
            "device": str(device),
            "batch_size": args.batch_size,
            "threads": args.threads,
            "window_count": args.window_count,
            "window_bytes": args.window_bytes,
            "window_starts": starts.tolist(),
            **inspect(model, windows, args.batch_size, device, args.min_age),
        }
        with _atomic_output(args.json, "w", overwrite=args.force) as out:
            json.dump(record, out, indent=2)
            out.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    print(f"{record['old_probes']:,} of {record['probe_events']:,} probes had phase age >= {args.min_age} bytes")


if __name__ == "__main__":
    main()
