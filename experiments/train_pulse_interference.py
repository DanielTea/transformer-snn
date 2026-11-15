"""Test whether per-cell write holds resist interference without blocking valid rewrites."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from common import Timer, _atomic_output, cosine_lr, make_optimizer, resolve_device
from train_pulse_recall import DISTRACTOR, QUERY, WRITE, PulseRecall, aggregate
from snnformer.pulse_memory import PulseMemory


@dataclass
class InterferenceBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    rapid_overwrite: torch.Tensor
    final_position: torch.Tensor
    rival_reads: torch.Tensor


def make_batch(size, keys, classes, min_gap, max_gap, generator, *, competing=True):
    """Give every write the same operation token, with independent values."""
    rows = torch.arange(size)
    focal_key = torch.randint(keys, (size,), generator=generator)
    old_value = torch.randint(classes, (size,), generator=generator)
    replacement = torch.randint(classes - 1, (size,), generator=generator)
    replacement += (replacement >= old_value).long()
    rapid_overwrite = torch.rand(size, generator=generator) < 0.5
    target_value = torch.where(rapid_overwrite, replacement, old_value)
    final_position = 3 + torch.randint(min_gap, max_gap + 1, (size,), generator=generator)

    shape = (size, max_gap + 4)
    op = torch.full(shape, DISTRACTOR, dtype=torch.long)
    key = torch.randint(keys, shape, generator=generator)
    value = torch.randint(classes, shape, generator=generator)
    targets = torch.full(shape, -100, dtype=torch.long)
    rival_reads = torch.zeros(shape, dtype=torch.bool)

    other_key = torch.randint(keys - 1, (size,), generator=generator)
    other_key += (other_key >= focal_key).long()
    other_value = torch.randint(classes, (size,), generator=generator)
    op[:, :2] = WRITE
    key[:, 0], value[:, 0] = focal_key, old_value
    key[:, 1] = torch.where(rapid_overwrite, focal_key, other_key)
    value[:, 1] = torch.where(rapid_overwrite, replacement, other_value)

    # The early query tests the response to an adjacent write. The final
    # query asks for that same focal value after the rest of the episode.
    for position in (2, final_position):
        op[rows, position] = QUERY
        key[rows, position] = focal_key
        value[rows, position] = classes
        targets[rows, position] = target_value

    # Each complete pair of intervening tokens writes and then queries a
    # rival key. Both the focal and the rival answers count towards the loss.
    pair_starts = torch.arange(max_gap // 2) * 2 + 3
    rival_key = torch.randint(keys - 1, (size, len(pair_starts)), generator=generator)
    rival_key += (rival_key >= focal_key[:, None]).long()
    rival_value = torch.randint(classes, rival_key.shape, generator=generator)
    if competing:
        pair_rows, pair_columns = torch.where((pair_starts + 1) < final_position[:, None])
        write_at = pair_starts[pair_columns]
        query_at = write_at + 1
        op[pair_rows, write_at] = WRITE
        op[pair_rows, query_at] = QUERY
        key[pair_rows, write_at] = key[pair_rows, query_at] = rival_key[pair_rows, pair_columns]
        value[pair_rows, write_at] = targets[pair_rows, query_at] = rival_value[pair_rows, pair_columns]
        value[pair_rows, query_at] = classes
        rival_reads[pair_rows, query_at] = True
    return InterferenceBatch(torch.stack((op, key, value), dim=-1), targets, rapid_overwrite,
                             final_position, rival_reads)


class PulseInterference(PulseRecall):
    """Start both arms with identical learned-parameter shapes and values."""

    def __init__(self, args, write_holds):
        super().__init__(args, carry_reads=True)
        if args.read_bins == 4:
            self.memory = PulseMemory(args.dim, args.cells, spike_cap=args.spike_cap,
                                      write_holds=write_holds, read_bins=args.read_bins,
                                      fixed_cuts=args.fixed_cuts, global_cuts=args.global_cuts)
            self.head = nn.Linear(self.memory.read_width, args.classes)
        elif write_holds is not None:
            protected = PulseMemory(args.dim, args.cells, spike_cap=args.spike_cap, write_holds=write_holds)
            protected.load_state_dict(self.memory.state_dict())
            self.memory = protected


def train(args, device, seed, arm, write_holds):
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = PulseInterference(args, write_holds).to(device)
    optimizer = make_optimizer(model, args.lr, args.wd)
    timer, curve = Timer(), []
    tokens, targets = 0, 0
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, args.steps, args.warmup, args.lr)
        batch = make_batch(args.bs, args.keys, args.classes, 0, args.train_max_gap, generator)
        truth = batch.targets.to(device)
        logits = model(batch.tokens.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), truth.flatten(), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += (batch.final_position + 1).sum().item()
        targets += (batch.targets != -100).sum().item()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            curve.append({"step": step + 1, "train_loss": loss.item()})
            print(f"{arm} seed {seed} step {step + 1} loss {loss.item():.3f} ({timer():.1f}s)", flush=True)

    report = {
        "params": sum(p.numel() for p in model.parameters()),
        "training": {"tokens": tokens, "scored_targets": targets, "curve": curve, "seconds": timer()},
    }
    if args.global_cuts:
        with torch.inference_mode():
            spikes = torch.zeros(1, args.dim, device=device)
            _, cuts, _, _ = model.memory._events(spikes, None, None)
            report["learned_global_cuts"] = cuts[0].int().tolist()
    return model, report


def score(correct, count):
    return {"correct": correct, "samples": count, "accuracy": correct / count if count else None}


@torch.inference_mode()
def evaluate(model, args, device, gap, competing, seed, samples):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    names = ("rapid_overwrite", "early_after_rival_write", "final_after_overwrite",
             "final_without_overwrite", "final", "rival_read")
    correct, counts = dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    events = {"write_tokens": 0}
    for offset in range(0, samples, args.eval_bs):
        size = min(args.eval_bs, samples - offset)
        batch = make_batch(size, args.keys, args.classes, gap, gap, generator, competing=competing)
        events["write_tokens"] += (batch.tokens[:, :, 0] == WRITE).sum().item()
        stats = {}
        pred = model(batch.tokens.to(device), stats=stats).argmax(-1)
        hits = pred == batch.targets.to(device)
        rapid = batch.rapid_overwrite.to(device)
        early = hits[:, 2]
        final = hits.gather(1, batch.final_position.to(device)[:, None]).squeeze(1)
        measures = (early[rapid], early[~rapid], final[rapid], final[~rapid], final,
                    hits[batch.rival_reads.to(device)])
        for name, measure in zip(names, measures):
            correct[name] += measure.sum().item()
            counts[name] += measure.numel()
        read_bins = stats.pop("read_bin_counts", None)
        if read_bins is not None:
            for read_bin, count in enumerate(read_bins.tolist()):
                name = f"read_bin_{read_bin}_events"
                events[name] = events.get(name, 0) + int(count)
        for name, value in stats.items():
            events[name] = events.get(name, 0) + int(value.item())

    return {"gap": gap, "competing": competing,
            "scores": {name: score(correct[name], counts[name]) for name in names}, "events": events}


@torch.inference_mode()
def match_rate(model, protected, args, device):
    """Shift the unrestricted arm's write bias using separate calibration episodes."""
    def record(candidate):
        return evaluate(candidate, args, device, args.train_max_gap, True, args.calibration_seed, args.eval_bs * 4)

    target = record(protected)["events"]["write_events"]
    write_bias = model.memory.bias[2]
    original = write_bias.clone()
    low, high = -32.0, 32.0
    best_shift, best_count = 0.0, record(model)["events"]["write_events"]
    for _ in range(14):
        shift = (low + high) / 2
        write_bias.copy_(original + shift)
        count = record(model)["events"]["write_events"]
        if abs(count - target) < abs(best_count - target):
            best_shift, best_count = shift, count
        if count < target:
            low = shift
        else:
            high = shift
    write_bias.copy_(original + best_shift)
    return {"bias_shift": best_shift, "matched_writes": best_count, "target_writes": target,
            "episodes": args.eval_bs * 4, "gap": args.train_max_gap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=16)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--cells", type=int, default=96)
    parser.add_argument("--spike-cap", type=int, default=32)
    parser.add_argument("--read-bins", type=int, choices=(2, 4), default=2,
                        help="two complementary pulses or four pulses with learned cuts")
    cut_mode = parser.add_mutually_exclusive_group()
    cut_mode.add_argument("--fixed-cuts", action="store_true", help="use fixed phase cuts with four read bins")
    cut_mode.add_argument("--global-cuts", action="store_true",
                          help="learn query-independent phase cuts for each cell with four read bins")
    parser.add_argument("--holds", type=int, nargs="+", default=[0, 4, 16, 64],
                        help="hold periods cycled across cells")
    parser.add_argument("--train-max-gap", type=int, default=16)
    parser.add_argument("--eval-gaps", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--eval-bs", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seed", type=int, default=20261115)
    parser.add_argument("--calibration-seed", type=int, default=20261116)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--no-rate-match", action="store_true")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if min(args.keys, args.classes) < 2 or min(args.dim, args.cells, args.spike_cap, args.steps, args.bs,
                                                args.eval_bs, args.eval_samples, args.log_every, args.threads) < 1:
        parser.error("at least two keys and classes and positive model, batch and training sizes are required")
    if min(args.train_max_gap, *args.eval_gaps, args.warmup, *args.holds) < 0 or max(args.holds) < 2:
        parser.error("gaps, warmup and holds must be nonnegative, and at least one hold must exceed one")
    if len(args.holds) > args.cells:
        parser.error("there must be at least as many cells as requested hold durations")
    if args.fixed_cuts and args.read_bins != 4:
        parser.error("--fixed-cuts requires --read-bins 4")
    if args.global_cuts and args.read_bins != 4:
        parser.error("--global-cuts requires --read-bins 4")
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.wd) and args.wd >= 0):
        parser.error("learning rate must be positive and weight decay nonnegative, both finite")
    seeds = args.seeds + [args.eval_seed, args.calibration_seed]
    if min(seeds) < 0 or max(seeds) >= 2**64 or len(set(seeds)) != len(seeds):
        parser.error("use distinct unsigned 64-bit training, evaluation and calibration seeds")
    path = args.json.expanduser()
    if path.exists() and not args.force:
        parser.error(f"{path} already exists, choose a new path or use --force")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)

    def score_all(model):
        return [evaluate(model, args, device, gap, competing, args.eval_seed, args.eval_samples)
                for gap in sorted(set(args.eval_gaps)) for competing in (True, False)]

    write_holds = tuple(args.holds[cell % len(args.holds)] for cell in range(args.cells))
    runs = []
    for seed in args.seeds:
        baseline, original = train(args, device, seed, "unrestricted", None)
        protected, held = train(args, device, seed, "protected", write_holds)
        if original["params"] != held["params"]:
            raise ValueError("the unrestricted and protected arms must have equal parameter counts")
        original["evaluation"], held["evaluation"] = score_all(baseline), score_all(protected)
        arms = {"unrestricted": original, "protected": held}
        if not args.no_rate_match:
            calibration = match_rate(baseline, protected, args, device)
            arms["rate_matched"] = {"calibration": calibration, "evaluation": score_all(baseline)}
        runs.append({"seed": seed, "arms": arms})
        for arm, report in arms.items():
            measures = " ".join(f"{row['gap']}={row['scores']['final']['accuracy']:.3f}"
                                for row in report["evaluation"] if row["competing"])
            print(f"{arm} seed {seed} final accuracy with competing writes {measures}", flush=True)

    summary = aggregate(runs)
    for arm, rows in summary.items():
        for row, item in zip(rows, runs[0]["arms"][arm]["evaluation"]):
            row["competing"] = item["competing"]
    settings = {key: value for key, value in vars(args).items() if key not in ("json", "force")}
    if args.read_bins == 2:  # Keep the existing two-pulse artifact schema.
        settings.pop("read_bins")
    if not args.fixed_cuts:
        settings.pop("fixed_cuts")
    if not args.global_cuts:
        settings.pop("global_cuts")
    payload = {
        "args": settings,
        "device": str(device), "chance_accuracy": 1 / args.classes, "write_holds": write_holds,
        "gap_definition": "tokens between immediate and final focal queries; each complete pair writes and queries a rival",
        "runs": runs, "aggregate": summary,
    }
    with _atomic_output(path, "w", args.force) as destination:
        json.dump(payload, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
