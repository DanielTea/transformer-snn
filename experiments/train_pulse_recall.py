"""Train pulse memory on repeated reads and overwrites of random bindings."""

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
from snnformer.config import SNNConfig
from snnformer.pulse_memory import PulseMemory
from snnformer.temporal_snn import SpikingPopulation


DISTRACTOR, WRITE, QUERY = range(3)
READ_STAGES = ("first_read", "repeated_read", "first_after_overwrite", "final_read")


@dataclass
class RecallBatch:
    tokens: torch.Tensor
    targets: torch.Tensor
    focal_reads: torch.Tensor
    other_reads: torch.Tensor
    lengths: torch.Tensor


def make_batch(size, keys, pairs, classes, min_gap, max_gap, generator):
    """Construct one episode per row with independent labels and no answer on queries."""
    rows = torch.arange(size)
    bindings = torch.rand(size, keys, generator=generator).argsort(dim=1)[:, :pairs]
    values = torch.randint(classes, (size, pairs), generator=generator)
    focal_index = torch.randint(pairs, (size,), generator=generator)
    focal_keys = bindings[rows, focal_index]
    replacement = torch.randint(classes, (size,), generator=generator)
    gaps = torch.randint(min_gap, max_gap + 1, (size,), generator=generator)

    # Each gap region contains distractors as well as queries of other stored keys.
    # The fixed other-key query before the first gap is also present when gap is zero.
    first = pairs + 1 + gaps
    repeated = first + gaps + 1
    overwrite = repeated + 1
    after = overwrite + 1
    final = after + gaps + 1
    focal_reads = torch.stack((first, repeated, after, final), dim=1)
    positions = torch.arange(pairs + 3 * max_gap + 6)
    shape = (size, len(positions))
    in_gap = ((positions >= pairs + 1) & (positions < first[:, None]))
    in_gap |= (positions > first[:, None]) & (positions < repeated[:, None])
    in_gap |= (positions > after[:, None]) & (positions < final[:, None])

    other_index = torch.randint(pairs - 1, shape, generator=generator)
    other_index += (other_index >= focal_index[:, None]).long()
    other_reads = in_gap & (torch.rand(shape, generator=generator) < 0.5)
    other_reads[:, pairs] = True

    op = torch.full(shape, DISTRACTOR, dtype=torch.long)
    key = torch.randint(keys, shape, generator=generator)
    value = torch.randint(classes, shape, generator=generator)
    target = torch.full(shape, -100, dtype=torch.long)
    op[:, :pairs] = WRITE
    key[:, :pairs] = bindings
    value[:, :pairs] = values
    op[other_reads] = QUERY
    key[other_reads] = bindings.gather(1, other_index)[other_reads]
    value[other_reads] = classes  # fixed sentinel, shared by all queries
    target[other_reads] = values.gather(1, other_index)[other_reads]
    op[rows, overwrite] = WRITE
    key[rows, overwrite] = focal_keys
    value[rows, overwrite] = replacement
    op[rows[:, None], focal_reads] = QUERY
    key[rows[:, None], focal_reads] = focal_keys[:, None]
    value[rows[:, None], focal_reads] = classes
    old = values[rows, focal_index]
    target[rows[:, None], focal_reads] = torch.stack((old, old, replacement, replacement), dim=1)
    return RecallBatch(torch.stack((op, key, value), dim=-1), target, focal_reads, other_reads, final + 1)


class PulseRecall(nn.Module):
    """Factorized token embeddings, then spiking memory as the only route to logits."""

    def __init__(self, args, carry_reads):
        super().__init__()
        self.operation = nn.Embedding(3, args.dim)
        self.key = nn.Embedding(args.keys, args.dim)
        self.value = nn.Embedding(args.classes + 1, args.dim)
        cfg = SNNConfig(neuron="ilif", levels=args.spike_cap, temporal_carry=False)
        self.input_emit = SpikingPopulation(args.dim, cfg, min(4.0, args.spike_cap / 2))
        self.memory = PulseMemory(args.dim, args.cells, carry_reads=carry_reads, spike_cap=args.spike_cap)
        self.head = nn.Linear(args.cells, args.classes)

    def forward(self, tokens, *, clear=None, stats=None):
        operation, key, value = tokens.unbind(-1)
        currents = self.operation(operation) + self.key(key) + self.value(value)
        spikes = self.input_emit(currents)
        read = self.memory(spikes, probe=(operation == QUERY), write=(operation == WRITE), clear=clear, stats=stats)
        return self.head(read)


@torch.inference_mode()
def evaluate(model, args, device, gap, *, clear_final=False):
    model.eval()
    generator = torch.Generator().manual_seed(args.eval_seed)
    correct_by_stage = torch.zeros(len(READ_STAGES), dtype=torch.long, device=device)
    changed_correct = torch.zeros(2, dtype=torch.long, device=device)
    changed_total = 0
    other_correct, other_total = 0, 0
    events = {}
    for start in range(0, args.eval_samples, args.eval_bs):
        size = min(args.eval_bs, args.eval_samples - start)
        batch = make_batch(size, args.keys, args.pairs, args.classes, gap, gap, generator)
        focal = batch.focal_reads.to(device)
        tokens = batch.tokens.to(device)
        targets = batch.targets.to(device)
        clear = None
        if clear_final:
            clear = torch.zeros_like(targets, dtype=torch.bool).scatter_(1, focal[:, -1:], True)
        stats = {}
        hits = model(tokens, clear=clear, stats=stats).argmax(-1) == targets
        focal_hits = hits.gather(1, focal)
        correct_by_stage += focal_hits.sum(dim=0)
        focal_targets = targets.gather(1, focal)
        changed = focal_targets[:, 0] != focal_targets[:, -1]
        changed_correct += (focal_hits[:, -2:] & changed[:, None]).sum(dim=0)
        changed_total += changed.sum().item()
        other_correct += (hits & batch.other_reads.to(device)).sum().item()
        other_total += batch.other_reads.sum().item()
        for name, tally in stats.items():
            events[name] = events.get(name, 0) + int(tally.item())

    def row(correct, count):
        return {"correct": correct, "samples": count, "accuracy": correct / count if count else None}

    scores = {stage: row(correct, args.eval_samples)
              for stage, correct in zip(READ_STAGES, correct_by_stage.tolist())}
    scores["other_keys"] = row(other_correct, other_total)
    scores["changed_overwrite_first_read"], scores["changed_overwrite_final_read"] = (
        row(correct, changed_total) for correct in changed_correct.tolist()
    )
    return {"gap": gap, "scores": scores, "events": events}


def train(args, device, seed, carry_reads):
    torch.manual_seed(seed)
    model = PulseRecall(args, carry_reads).to(device)
    optimizer = make_optimizer(model, args.lr, args.wd)
    generator = torch.Generator().manual_seed(seed)
    timer, curve = Timer(), []
    tokens, padded_positions, targets = 0, 0, 0
    arm = "preserve" if carry_reads else "erase"
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, args.steps, args.warmup, args.lr)
        batch = make_batch(args.bs, args.keys, args.pairs, args.classes, 0, args.train_max_gap, generator)
        label = batch.targets.to(device)
        logits = model(batch.tokens.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), label.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tokens += batch.lengths.sum().item()
        padded_positions += batch.targets.numel()
        targets += (batch.targets >= 0).sum().item()
        if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
            curve.append({"step": step + 1, "train_loss": loss.item()})
            print(f"{arm} seed {seed} step {step + 1} loss {loss.item():.3f} ({timer():.1f}s)", flush=True)

    return model, {
        "params": sum(parameter.numel() for parameter in model.parameters()),
        "training": {"steps": args.steps, "episodes": args.steps * args.bs, "tokens": tokens,
                     "batched_positions": padded_positions, "scored_targets": targets,
                     "seconds": timer(), "curve": curve},
    }


def aggregate(runs):
    """Pool counts while retaining each seed's scores for assessing variation."""
    summary = {}
    for arm in runs[0]["arms"]:
        rows = []
        evaluations = [run["arms"][arm]["evaluation"] for run in runs]
        for points in zip(*evaluations):
            scores = {}
            for stage in points[0]["scores"]:
                correct = sum(point["scores"][stage]["correct"] for point in points)
                samples = sum(point["scores"][stage]["samples"] for point in points)
                scores[stage] = {"correct": correct, "samples": samples,
                                 "accuracy": correct / samples if samples else None}
            events = {name: sum(point["events"][name] for point in points) for name in points[0]["events"]}
            rows.append({"gap": points[0]["gap"], "scores": scores, "events": events})
        summary[arm] = rows
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", type=int, default=16)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--classes", type=int, default=4)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--cells", type=int, default=96)
    parser.add_argument("--spike-cap", type=int, default=32)
    parser.add_argument("--train-max-gap", type=int, default=8)
    parser.add_argument("--eval-gaps", type=int, nargs="+", default=[8, 32, 128, 512])
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--eval-bs", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=512, help="episodes per gap and arm")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-seed", type=int, default=20261115)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, required=True, help="destination for the reproducible result")
    parser.add_argument("--force", action="store_true", help="replace an existing JSON result")
    args = parser.parse_args()

    if not (2 <= args.pairs <= args.keys and args.classes >= 2):
        parser.error("require 2 <= --pairs <= --keys and at least two classes")
    if min(args.dim, args.cells, args.spike_cap, args.steps, args.bs, args.eval_bs, args.eval_samples,
           args.log_every, args.threads) < 1:
        parser.error("model, batch, training, evaluation, and thread sizes must be positive")
    if args.train_max_gap < 0 or min(args.eval_gaps) < 0 or args.warmup < 0:
        parser.error("gaps and warmup must be nonnegative")
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.wd) and args.wd >= 0):
        parser.error("learning rate must be positive and weight decay nonnegative, both finite")
    seeds = args.seeds + [args.eval_seed]
    if min(seeds) < 0 or max(seeds) >= 2**64 or len(set(seeds)) != len(seeds):
        parser.error("use distinct unsigned 64-bit seeds for training and evaluation")
    path = args.json.expanduser()
    if path.exists() and not args.force:
        parser.error(f"{path} already exists, choose a new path or use --force")
    try:
        device = resolve_device(args.device)
    except ValueError as exc:
        parser.error(str(exc))
    torch.set_num_threads(args.threads)

    def score_gaps(model, clear_final=False):
        return [evaluate(model, args, device, gap, clear_final=clear_final) for gap in sorted(set(args.eval_gaps))]

    reports = []
    for seed in args.seeds:
        arms = {}
        for arm in ("preserve", "erase"):
            model, report = train(args, device, seed, carry_reads=(arm == "preserve"))
            report["evaluation"] = score_gaps(model)
            arms[arm] = report
            if arm == "preserve":
                arms["clear_before_final"] = {"evaluation": score_gaps(model, clear_final=True)}
        reports.append({"seed": seed, "arms": arms})
        for arm, report in arms.items():
            scores = " ".join(f"gap {point['gap']}={point['scores']['final_read']['accuracy']:.3f}"
                              for point in report["evaluation"])
            print(f"{arm} seed {seed} final read {scores}", flush=True)

    payload = {
        "args": {key: value for key, value in vars(args).items() if key not in ("json", "force")},
        "device": str(device), "chance_accuracy": 1 / args.classes,
        "gap_definition": "intervening tokens before the first, repeated, and final focal reads",
        "runs": reports, "aggregate": aggregate(reports),
    }
    with _atomic_output(path, "w", args.force) as destination:
        json.dump(payload, destination, indent=2)
        destination.write("\n")


if __name__ == "__main__":
    main()
