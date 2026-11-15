"""Measure delayed next-byte effects of replacing single accepted membrane writes."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import Timer, _atomic_output, resolve_device
from probe_lm_context import load_model
from train_lm import DATA, ENWIK8_SHA1, load_splits


def windows_at(data, starts, context):
    """Slice context input bytes plus one extra byte for next-byte targets."""
    offsets = np.arange(context + 1, dtype=np.int64)
    return torch.from_numpy(data[np.asarray(starts)[:, None] + offsets].astype(np.int64))


def sample_training_windows(data, batch_size, context, generator):
    """Sample [batch, context + 1] windows uniformly from a supplied training split."""
    if len(data) < context + 1:
        raise ValueError("the data split must contain the requested context plus a target byte")
    starts = torch.randint(0, len(data) - context, (batch_size,), generator=generator)
    return windows_at(data, starts.numpy(), context)


def sample_protected_writes(accepted_writes, write_holds, max_age, generator, draws=1):
    """Sample [draws, batch] accepted protected write positions and cells with replacement.

    Every sampled write has the full requested delay available in the input
    window. A stream with no candidate gets a pair of -1 sentinels.
    """
    batch, context, cells = accepted_writes.shape
    if max_age < 1 or max_age >= context or draws < 1 or len(write_holds) != cells:
        raise ValueError("write sampling requires valid holds, horizon and draw count")
    protected = torch.as_tensor(write_holds, device=accepted_writes.device) > 0
    eligible = (accepted_writes[:, :context - max_age].bool() & protected).cpu()
    positions = torch.full((draws, batch), -1, dtype=torch.long)
    selected_cells = positions.clone()
    for row, mask in enumerate(eligible):
        choices = mask.nonzero(as_tuple=False)
        if len(choices):
            chosen = choices[torch.randint(len(choices), (draws,), generator=generator)]
            positions[:, row], selected_cells[:, row] = chosen.unbind(dim=-1)
    return positions.to(accepted_writes.device), selected_cells.to(accepted_writes.device)


@dataclass
class DelayedScores:
    deltas: torch.Tensor
    valid: torch.Tensor
    eligible: torch.Tensor
    changed_read: torch.Tensor


@torch.no_grad()
def score_write_delays(model, trace, targets, factual_logits, write_positions, write_cells, min_age, max_age):
    """Paired BPB effect at every delay for one chosen write per input row."""
    memory = model.blocks[-1].memory
    counterfactual_read, exposed = memory.counterfactual_read(
        trace, write_positions, write_cells, min_age=min_age, max_age=max_age,
    )
    valid_writes = write_positions >= 0
    offset = torch.arange(min_age, max_age + 1, device=targets.device)
    deadlines = write_positions.clamp_min(0)[:, None] + offset
    rows = torch.arange(targets.shape[0], device=targets.device)[:, None]
    factual_read = trace["reads"][rows, deadlines]
    replacement_read = counterfactual_read[rows, deadlines]
    valid = valid_writes[:, None].expand_as(deadlines)
    eligible = exposed[rows, deadlines] & valid
    changed_read = (factual_read != replacement_read).any(dim=-1) & valid
    if (changed_read & ~eligible).any():
        raise RuntimeError("read replacement affected a target outside the write's applicable probes")
    deltas = torch.zeros_like(deadlines, dtype=torch.float32)

    # Without a changed read, the final-block decoder gives exactly the same
    # logits. Only replay positions with an actual change in a routed event.
    if changed_read.any():
        old_current = trace["pre_memory_current"][rows, deadlines][changed_read]
        prior_logits = model.replay_memory_read(old_current, replacement_read[changed_read])
        original = factual_logits[rows, deadlines][changed_read]
        expected = targets[rows, deadlines][changed_read]
        prior_loss = F.cross_entropy(prior_logits, expected, reduction="none")
        factual_loss = F.cross_entropy(original, expected, reduction="none")
        deltas[changed_read] = (prior_loss - factual_loss) / math.log(2)
    return DelayedScores(deltas, valid, eligible, changed_read)


def signed_summary(batches, tolerance):
    values = torch.cat(batches).numpy().astype(np.float64)
    if not len(values):
        return {"count": 0}
    if not np.isfinite(values).all():
        raise RuntimeError("a sampled write produced a nonfinite loss difference")
    positive = int(np.count_nonzero(values > tolerance))
    negative = int(np.count_nonzero(values < -tolerance))
    return {
        "count": len(values), "mean_bpb": float(values.mean()),
        "mean_absolute_bpb": float(np.abs(values).mean()),
        "quantiles_bpb": {str(q): float(np.quantile(values, q)) for q in (0, 0.01, 0.1, 0.5, 0.9, 0.99, 1)},
        "positive_count": positive, "negative_count": negative,
        "within_tolerance_count": len(values) - positive - negative,
        "positive_fraction": positive / len(values), "negative_fraction": negative / len(values),
    }


@torch.inference_mode()
def inspect(model, windows, args, device):
    if model.cfg.temporal_carry or not model.cfg.pulse_preserve_reads or not any(model.cfg.pulse_write_holds):
        raise ValueError("expected a nondestructive bank with protected writes and token-local output neurons")
    generator = torch.Generator().manual_seed(args.selection_seed)
    totals = Counter()
    by_cell = Counter()
    target_buckets = {"all_delays": [], "eligible_and_probed": [], "changed_read": []}
    write_buckets = {"all_delays": [], "eligible_and_probed": []}
    timer = Timer()
    for number, windows_batch in enumerate(windows.split(args.batch_size)):
        batch = windows_batch.to(device)
        targets = batch[:, 1:]
        trace = {}
        logits = model(batch[:, :-1], memory_trace=trace)
        if number == 0:
            replayed = model.replay_memory_read(trace["pre_memory_current"], trace["reads"])
            if not torch.equal(replayed, logits):
                raise RuntimeError("factual replay does not reproduce the original model logits exactly")
        draws = sample_protected_writes(trace["accepted_writes"], model.cfg.pulse_write_holds,
                                        args.max_age, generator, args.draws)
        positions, cells = (tensor.cpu() for tensor in draws)
        totals["candidate_draws"] += positions.numel()
        totals["candidate_draws_without_eligible_write"] += int((positions < 0).sum())
        for row in range(len(batch)):
            distinct = set(zip(positions[:, row].tolist(), cells[:, row].tolist()))
            totals["distinct_writes_sampled"] += len(distinct - {(-1, -1)})
        by_cell.update(cell for cell in cells.flatten().tolist() if cell >= 0)

        for write_positions, write_cells in zip(*draws):
            effect = score_write_delays(model, trace, targets, logits, write_positions, write_cells,
                                        args.min_age, args.max_age)
            deltas, valid, eligible, changed = (tensor.cpu() for tensor in (
                effect.deltas, effect.valid, effect.eligible, effect.changed_read,
            ))
            valid_rows = valid.any(dim=1)
            eligible_rows = eligible.any(dim=1)
            totals["sampled_writes"] += int(valid_rows.sum())
            totals["writes_with_eligible_target"] += int(eligible_rows.sum())
            totals["writes_with_changed_read"] += int(changed.any(dim=1).sum())
            totals["writes_with_loss_change"] += int((deltas.abs() > args.tolerance).any(dim=1).sum())
            target_buckets["all_delays"].append(deltas[valid])
            target_buckets["eligible_and_probed"].append(deltas[eligible])
            target_buckets["changed_read"].append(deltas[changed])
            write_buckets["all_delays"].append(deltas[valid_rows].mean(dim=1))
            counts = eligible[eligible_rows].sum(dim=1)
            write_buckets["eligible_and_probed"].append(deltas[eligible_rows].sum(dim=1) / counts)

    trials = totals["sampled_writes"]
    return {
        "counts": dict(totals),
        "candidate_draws_by_cell": dict(sorted(by_cell.items())),
        "write_prevalence": {name: totals[name] / trials if trials else None for name in (
            "writes_with_eligible_target", "writes_with_changed_read", "writes_with_loss_change",
        )},
        "target_level_effect_bpb": {name: signed_summary(values, args.tolerance)
                                    for name, values in target_buckets.items()},
        "per_write_mean_effect_bpb": {name: signed_summary(values, args.tolerance)
                                     for name, values in write_buckets.items()},
        "factual_replay_exact": True,
        "seconds": timer(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", DATA)))
    parser.add_argument("--split", choices=("both", "val", "test"), default="both")
    parser.add_argument("--window-count", type=int, default=128)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-age", type=int, default=128)
    parser.add_argument("--max-age", type=int, default=512)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--selection-seed", type=int, default=20261115)
    parser.add_argument("--tolerance", type=float, default=1e-6, help="zero threshold for signed counts in BPB")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing JSON result")
    args = parser.parse_args()
    if min(args.window_count, args.context, args.batch_size, args.threads, args.draws) < 1:
        parser.error("window, batch, thread and draw sizes must be positive")
    if not 1 <= args.min_age <= args.max_age < args.context:
        parser.error("the delay band must be positive, ordered and fit the context")
    if not math.isfinite(args.tolerance) or args.tolerance < 0 or not 0 <= args.selection_seed < 2**64:
        parser.error("the tolerance must be finite and nonnegative and the seed must be an unsigned 64-bit integer")
    path = args.json.expanduser()
    if path.exists() and not args.force:
        parser.error(f"{path} already exists, choose a new path or use --force")

    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        _, val, test = load_splits(args.data.expanduser())
        split_names = ("val", "test") if args.split == "both" else (args.split,)
        windows = {}
        starts_by_split = {}
        for name in split_names:
            data = val if name == "val" else test
            width = args.context + 1
            if width * args.window_count > len(data):
                raise ValueError(f"the requested {name} windows would overlap or exceed the split")
            starts = np.linspace(0, len(data) - width, args.window_count, dtype=np.int64)
            windows[name] = windows_at(data, starts, args.context)
            starts_by_split[name] = starts.tolist()

        results = []
        for checkpoint in args.checkpoint:
            model, run_name = load_model(checkpoint.expanduser(), device)
            reports = {}
            for name in split_names:
                report = inspect(model, windows[name], args, device)
                reports[name] = report
                mean = report["per_write_mean_effect_bpb"]["all_delays"].get("mean_bpb")
                print(f"{run_name} {name} sampled {report['counts']['sampled_writes']:,} writes, "
                      f"mean delayed BPB effect {mean} ({report['seconds']:.1f}s)", flush=True)
            results.append({"run_name": run_name, "checkpoint": str(checkpoint), "splits": reports})
        record = {
            "data_sha1": ENWIK8_SHA1, "device": str(device), "threads": args.threads, "batch_size": args.batch_size,
            "context": args.context, "window_starts": starts_by_split, "selection_seed": args.selection_seed,
            "draws_per_window": args.draws, "signed_tolerance_bpb": args.tolerance,
            "read_delay_from_write": [args.min_age, args.max_age],
            "selection": "uniform with replacement among accepted protected cell writes with the full horizon available",
            "eligibility": "the selected write still supplies a live phase and its cell is probed at this delay",
            "effect": "counterfactual BPB minus factual BPB; the counterfactual preserves the write schedule and "
                      "replaces only the selected write payload with its preceding phase",
            "models": results,
        }
        with _atomic_output(path, "w", args.force) as out:
            json.dump(record, out, indent=2, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
