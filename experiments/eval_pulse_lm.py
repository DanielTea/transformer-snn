"""Score matched byte targets and intervene on paired-pulse membrane state."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from benchmark_decode import state_bytes
from common import resolve_device
from probe_lm_context import load_model
from snnformer import count_params
from train_lm import DATA, ENWIK8_SHA1, load_splits


def remote_prefix_mask(windows, history, order, distance):
    """Flag targets whose most recent matching context occurs far back in the window."""
    remote = np.zeros((len(windows), windows.shape[1] - history), dtype=np.bool_)
    for row, window in enumerate(windows):
        last_seen = {}
        raw = window.tobytes()
        for end in range(order, len(raw)):
            prefix = raw[end - order:end]
            previous = last_seen.get(prefix)
            if end >= history and previous is not None:
                remote[row, end - history] = end - previous >= distance
            last_seen[prefix] = end
    return remote


@torch.inference_mode()
def score(model, windows, remote, history, batch_size, device, *, clear=False, old_phase_age=None):
    losses = torch.zeros(2, device=device)
    counters = {}
    write_counts = None
    for window, flagged in zip(windows.split(batch_size), remote.split(batch_size)):
        inputs = window[:, :-1].to(device)
        targets = window[:, history:].to(device)
        options = {}
        if model.cfg.pulse_memory_cells:
            options["memory_stats"] = {}
            if not clear and old_phase_age is None and model.cfg.charge_memory == "none":
                options["memory_training_stats"] = {}
        if clear:
            # Only clear the extra bank. The local attention stream is unchanged.
            clear_at = torch.zeros_like(inputs, dtype=torch.bool)
            clear_at[:, history - 1:] = True
            options["memory_clear_at"] = clear_at
        if old_phase_age is not None:
            options["memory_ablate_min_age"] = old_phase_age
        logits = model(inputs, **options)[:, history - 1:]
        for key, value in options.get("memory_stats", {}).items():
            counters[key] = counters.get(key, 0) + value
        if "memory_training_stats" in options:
            batch_counts = options["memory_training_stats"]["accepted_write_counts"]
            write_counts = batch_counts if write_counts is None else write_counts + batch_counts
        loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        losses += torch.stack((loss.sum(), (loss * flagged.to(device)).sum()))
    counts = (remote.numel(), int(remote.sum()))
    result = {
        "bpb": losses[0].item() / (counts[0] * math.log(2)),
        "remote_bpb": losses[1].item() / (counts[1] * math.log(2)) if counts[1] else None,
    }
    if counters:
        events = {key: int(value) if value.ndim == 0 else [int(x) for x in value.tolist()]
                  for key, value in counters.items()}
        inputs_seen = len(windows) * (windows.shape[1] - 1)
        result["memory_events"] = {
            **events,
            "restoration_spikes": events["probe_events"] - events["read_spikes"],
            "read_cycle_spikes_per_input_byte": events["probe_events"] / inputs_seen,
            "pulse_applications_per_input_byte": model.cfg.pulse_read_bins * events["probe_events"] / inputs_seen,
            "writes_per_input_byte": events["write_events"] / inputs_seen,
        }
        if write_counts is not None:
            result["memory_events"]["writes_per_input_byte_by_cell"] = (write_counts / inputs_seen).tolist()
    return result


@torch.inference_mode()
def evaluate(checkpoints, windows, remote, args, device):
    results = []
    for checkpoint in checkpoints:
        model, name = load_model(checkpoint, device)
        _, state = model.step(windows[:1, 0].to(device))
        row = {
            "run_name": name,
            "parameters": count_params(model),
            "stream_state_bytes": state_bytes(state),
            "full": score(model, windows, remote, args.history, args.batch_size, device),
        }
        if model.cfg.pulse_memory_cells:
            if args.old_phase_age is not None:
                row["old_phases_hidden_from_reads"] = score(model, windows, remote, args.history,
                                                           args.batch_size, device, old_phase_age=args.old_phase_age)
            else:
                row["bank_erased_at_targets"] = score(model, windows, remote, args.history,
                                                      args.batch_size, device, clear=True)
        print(f"{name} {json.dumps(row['full'])}", flush=True)
        results.append(row)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", DATA)))
    parser.add_argument("--window-count", type=int, default=1024)
    parser.add_argument("--history", type=int, default=1024)
    parser.add_argument("--tail", type=int, default=128)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--remote-distance", type=int, default=128)
    parser.add_argument("--old-phase-age", type=int,
                        help="hide phases at least this old from every memory read without changing state")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.window_count, args.history, args.tail, args.order, args.remote_distance,
           args.threads, args.batch_size) < 1 or args.order + args.remote_distance > args.history:
        parser.error("sizes must be positive, and history must cover the prefix and remote distance")
    if args.old_phase_age is not None and args.old_phase_age < 1:
        parser.error("--old-phase-age must be positive")
    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        _, _, test = load_splits(args.data.expanduser())
        width = args.history + args.tail
        if args.history + args.tail * args.window_count > len(test):
            parser.error("scored test tails would overlap or exceed the held-out split")
        starts = np.linspace(0, len(test) - width, args.window_count, dtype=np.int64)
        raw = test[starts[:, None] + np.arange(width)]
        remote = torch.from_numpy(remote_prefix_mask(raw, args.history, args.order, args.remote_distance))
        results = evaluate(args.checkpoint, torch.from_numpy(raw.astype(np.int64)), remote, args, device)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    record = {
        "data_sha1": ENWIK8_SHA1,
        "split": "test",
        "device": str(device),
        "history_bytes": args.history,
        "scored_bytes_per_window": args.tail,
        "window_starts": starts.tolist(),
        "target_bytes": remote.numel(),
        "prefix_bytes": args.order,
        "remote_min_distance_bytes": args.remote_distance,
        "remote_target_bytes": int(remote.sum()),
        "phase_ablation_age_bytes": args.old_phase_age,
        "models": results,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
