"""Fit the protected write gates using paired future loss from a frozen decoder."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from common import resolve_device, save_checkpoint, save_result
from inspect_write_credit import sample_protected_writes, score_write_delays
from probe_lm_context import load_model
from train_lm import DATA, ENWIK8_SHA1, evaluate_bpb, load_splits, random_batch


MIN_AGE, MAX_AGE = 128, 512
ARMS = ("ce", "paired", "shuffled")


@dataclass
class CreditBatch:
    x: torch.Tensor
    y: torch.Tensor
    spikes: torch.Tensor
    cells: torch.Tensor
    scores: torch.Tensor


@torch.no_grad()
def collect_credit(model, data, args, device):
    memory = model.blocks[-1].memory
    tokens = torch.Generator().manual_seed(args.seed + 131071)
    writes = torch.Generator().manual_seed(args.seed + 524287)
    batches = []
    for _ in range(args.steps):
        x, y = random_batch(data, args.batch_size, model.ctx, tokens)
        targets = y.to(device)
        trace = {}
        factual = model(x.to(device), memory_trace=trace)
        positions, cells = sample_protected_writes(trace["accepted_writes"], memory.write_holds,
                                                 MAX_AGE, writes, draws=args.draws)
        positions, cells = positions.to(device), cells.to(device)
        scores = []
        for position, cell in zip(positions, cells):
            delayed = score_write_delays(model, trace, targets, factual, position, cell, MIN_AGE, MAX_AGE)
            scores.append(delayed.deltas.mean(dim=1))
        valid = cells >= 0
        rows = torch.arange(x.shape[0], device=device).expand_as(cells)
        spikes = trace["spike_inputs"][rows[valid], positions[valid]]
        batches.append(CreditBatch(x, y, spikes.cpu(), cells[valid].cpu(), torch.stack(scores)[valid].cpu()))
    return batches


@torch.no_grad()
def calibration_drives(model, data, args, device):
    tokens = torch.Generator().manual_seed(104729)
    drives = []
    for _ in range(16):
        x, _ = random_batch(data, args.batch_size, model.ctx, tokens)
        stats = {}
        model(x.to(device), memory_training_stats=stats)
        drives.append(stats["write_drive"])
    return torch.cat(drives)


@torch.no_grad()
def match_accepted_rates(memory, drive, target):
    """Find per-cell bias offsets that match held write counts on training data."""
    lowest = -drive.amax(dim=(0, 1)) - 0.001
    highest = -drive.amin(dim=(0, 1)) + 0.001
    current = memory._accepted_writes((drive >= 0).float()).sum(dim=(0, 1))
    lowest = torch.where(current < target, 0, lowest)
    highest = torch.where(current > target, 0, highest)
    offset = torch.zeros_like(lowest)
    best = (current - target).abs()
    for _ in range(20):
        cutoff = (lowest + highest) / 2
        actual = memory._accepted_writes((drive + cutoff >= 0).float()).sum(dim=(0, 1))
        error = (actual - target).abs()
        preferable = (error < best) | ((error == best) & (cutoff.abs() < offset.abs()))
        offset = torch.where(preferable, cutoff, offset)
        best = torch.minimum(best, error)
        lowest = torch.where(actual < target, cutoff, lowest)
        highest = torch.where(actual >= target, cutoff, highest)
    memory.bias[2].add_(offset)
    return memory._accepted_writes((drive + offset >= 0).float()).sum(dim=(0, 1))


def fit_gate(model, batches, credits, scale, arm, args, device):
    memory = model.blocks[-1].memory
    model.requires_grad_(False)
    memory.project.weight.requires_grad_(True)
    memory.bias.requires_grad_(True)
    protected = torch.tensor(memory.write_holds, device=device) >= 128
    mask = torch.zeros_like(memory.bias, dtype=torch.bool)
    mask[2] = protected
    memory.project.weight.register_hook(lambda grad: grad * mask.flatten()[:, None])
    memory.bias.register_hook(lambda grad: grad * mask)
    optimizer = torch.optim.AdamW((memory.project.weight, memory.bias), lr=args.lr, weight_decay=0.)
    next_losses, write_losses = [], []
    for batch, scores in zip(batches, credits):
        logits = model(batch.x.to(device))
        next_loss = F.cross_entropy(logits.flatten(0, 1), batch.y.to(device).flatten())
        write_logits = memory.project(batch.spikes.to(device)).unflatten(-1, memory.bias.shape) + memory.bias
        selected = write_logits[:, 2].gather(1, batch.cells.to(device)[:, None]).squeeze(-1)
        scores = scores.to(device)
        weights = (scores.abs() / scale).clamp_max(1.)
        write_loss = F.binary_cross_entropy_with_logits(selected, (scores > 0).to(selected.dtype),
                                                        weight=weights, reduction="sum") / max(1, len(scores))
        loss = next_loss + (0. if arm == "ce" else args.credit_weight) * write_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        next_losses.append(next_loss.detach())
        write_losses.append(write_loss.detach())
    return torch.stack(next_losses).mean().item() / math.log(2), torch.stack(write_losses).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=Path(os.getenv("ENWIK8_PATH", str(DATA))))
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--credit-weight", type=float, default=0.05)
    parser.add_argument("--eval-batches", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.draws, args.threads, args.eval_batches, args.eval_batch_size) < 1:
        parser.error("steps, threads and sizes must be positive")
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.credit_weight) and args.credit_weight > 0):
        parser.error("learning rate and credit weight must be positive and finite")
    names = {arm: f"pulse_credit_{arm}_{args.tag}_s{args.seed}" for arm in ARMS}
    outputs = {arm: args.output_checkpoints / (name + ".pt") for arm, name in names.items()}
    if not args.force and any(path.exists() for path in outputs.values()):
        parser.error("output checkpoint exists; use --tag or --force")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if original["args"]["seed"] != args.seed:
        parser.error("seed must match the base checkpoint")
    base, name = load_model(args.checkpoint, device)
    memory = base.blocks[-1].memory
    if (memory is None or not memory.carry_reads or not memory.write_holds
            or min((hold for hold in memory.write_holds if hold), default=0) < MIN_AGE
            or base.cfg.temporal_carry or base.ctx <= MAX_AGE + 1):
        parser.error("expected a token-local decoder with protected, nondestructive pulse reads")
    if max(base.ctx * args.eval_batch_size * args.eval_batches, 1) + base.ctx >= 5_000_000:
        parser.error("evaluation windows exceed a held-out split")

    train, val, test = load_splits(args.data)
    calibration = calibration_drives(base, train, args, device)
    target = memory._accepted_writes((calibration >= 0).float()).sum(dim=(0, 1))
    calibration_total = calibration.shape[0] * calibration.shape[1]
    batches = collect_credit(base, train, args, device)
    del base, memory, calibration
    labels = torch.cat([batch.scores for batch in batches])
    cells = torch.cat([batch.cells for batch in batches])
    nonzero = labels[labels != 0]
    print(f"{name}: {len(nonzero)}/{len(labels)} sampled writes have nonzero paired credit", flush=True)
    if nonzero.numel() < 32 or not bool((nonzero > 0).any() and (nonzero < 0).any()):
        parser.error("too few informative paired scores of both signs for a matched training test")
    scale = nonzero.abs().quantile(0.95).item()
    shuffled = labels.clone()
    generator = torch.Generator().manual_seed(args.seed + 65537)
    for cell in cells.unique():
        indices = torch.where(cells == cell)[0]
        shuffled[indices] = labels[indices[torch.randperm(len(indices), generator=generator)]]
    sections = [len(batch.scores) for batch in batches]
    label_sets = {"paired": labels.split(sections), "ce": labels.split(sections), "shuffled": shuffled.split(sections)}

    serializable_args = argparse.Namespace(**{key: str(value) if isinstance(value, Path) else value
                                               for key, value in vars(args).items()})
    for arm in ARMS:
        model, _ = load_model(args.checkpoint, device)
        train_bpb, auxiliary = fit_gate(model, batches, label_sets[arm], scale, arm, args, device)
        drive = calibration_drives(model, train, args, device)
        actual = match_accepted_rates(model.blocks[-1].memory, drive, target)
        val_bpb = evaluate_bpb(model, val, model.ctx, args.eval_batches, args.eval_batch_size, device=device)
        test_bpb = evaluate_bpb(model, test, model.ctx, args.eval_batches, args.eval_batch_size, device=device)
        save_checkpoint(outputs[arm], model, original["model_kwargs"], args=serializable_args,
                        task="lm", name=names[arm])
        save_result("pulse_credit", names[arm], {
            "arm": arm, "base": name, "args": vars(serializable_args), "data_sha1": ENWIK8_SHA1,
            "train_bpb": train_bpb, "auxiliary_loss": auxiliary, "val_bpb": val_bpb, "test_bpb": test_bpb,
            "sampled_writes": len(labels), "nonzero_scores": len(nonzero),
            "positive_scores": int((labels > 0).sum()), "score_scale_bpb": scale,
            "target_rate_by_cell": (target / calibration_total).tolist(),
            "calibration_rate_by_cell": (actual / calibration_total).tolist(),
        }, overwrite=args.force)
        print(f"{names[arm]}: validation {val_bpb:.6f}, test {test_bpb:.6f} bpb", flush=True)


if __name__ == "__main__":
    main()
