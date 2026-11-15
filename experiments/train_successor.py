"""Calibrate a successor readout on a frozen local byte Transformer."""

from __future__ import annotations

import argparse
import math
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch

from common import checkpoint_path, make_lm, resolve_device, result_exists, save_checkpoint, save_result, setup
from snnformer import SNNConfig, count_params
from train_lm import DATA, ENWIK8_SHA1, load_splits, random_batch


def target_probability(log_probs, targets):
    return log_probs.gather(-1, targets[..., None]).squeeze(-1).exp()


def next_byte_nll(base_prob, correct, strength):
    added = (correct * strength).sum(-1)
    return strength.sum(-1).log1p() - (base_prob + added).clamp_min(torch.finfo(base_prob.dtype).tiny).log()


def evaluation_batches(data, args, device):
    context = args.eval_context
    full = getattr(args, "eval_full", False)
    if full:
        windows, stride = (len(data) - 1) // context, context
    else:
        windows = args.eval_windows
        stride = (len(data) - context - 1) // windows
    for start in range(0, windows, args.bs):
        indices = range(start, min(start + args.bs, windows))
        yield torch.stack([
            torch.tensor(data[i * stride : i * stride + context + 1], dtype=torch.long)
            for i in indices
        ]).to(device)
    if full and len(data) - windows * context > 1:
        yield torch.tensor(data[windows * context:], dtype=torch.long, device=device)[None, :]


@torch.no_grad()
def measure(model, analog, data, args, device):
    """Score all controls on exactly the same disjoint windows."""
    totals = torch.zeros(4, dtype=torch.float64)
    probs, choices, wins = [], [], []
    orders = model.readout.orders
    receptive_field = 1 + len(model.blocks) * (model.cfg.local_window - 1)
    hits, far_hits, far_wins, count = 0, 0, 0, 0
    for windows in evaluation_batches(data, args, device):
        tokens, targets = windows[:, :-1], windows[:, 1:]
        log_probs, successors, ages = model.evidence(tokens)
        base_prob = target_probability(log_probs, targets)
        correct = successors == targets[..., None]
        pulses = model.readout(log_probs, successors, ages, tokens)
        soft = pulses if analog is None else analog(log_probs, successors, ages, tokens, quantized=False)
        far = ages >= receptive_field
        nll = (-base_prob.log(), next_byte_nll(base_prob, correct, pulses),
               next_byte_nll(base_prob, correct, soft), next_byte_nll(base_prob, correct, pulses * far))
        totals += torch.tensor([term.sum().item() for term in nll], dtype=totals.dtype)
        hit = successors >= 0
        longest = torch.where(hit, orders, -1.).argmax(-1, keepdim=True)
        choices.append(torch.where(hit.any(-1), longest.squeeze(-1), -1).flatten().cpu())
        wins.append(correct.gather(-1, longest).flatten().cpu())
        probs.append(base_prob.flatten().cpu())
        hits += hit.any(-1).sum().item()
        far_hits += far.any(-1).sum().item()
        far_wins += (correct & far).any(-1).sum().item()
        count += targets.numel()
    metrics = dict(zip(("base_bpb", "spike_bpb", "analog_bpb", "remote_only_bpb"),
                       (totals / (count * math.log(2))).tolist()))
    if analog is None:
        del metrics["analog_bpb"]
    metrics.update(target_bytes=count, hit_rate=hits / count, remote_hit_rate=far_hits / count,
                   remote_correct_rate=far_wins / max(1, far_hits), remote_min_age=receptive_field)
    return metrics, (torch.cat(probs), torch.cat(choices), torch.cat(wins))


def fixed_cache_bpb(samples, lambdas):
    probs, choices, wins = samples
    weight = lambdas[choices.clamp_min(0)] * (choices >= 0)
    likelihood = (1 - weight) * probs + weight * wins
    return -likelihood.log2().mean().item()


def fit_fixed_cache(samples, orders):
    """Fit one interpolation coefficient per longest available order on validation."""
    probs, choices, wins = samples
    grid = torch.linspace(0.0, 0.95, 96)
    weights = torch.zeros(len(orders))
    for head in range(len(orders)):
        selected = choices == head
        if selected.any():
            sums = torch.zeros_like(grid, dtype=torch.float64)
            for p, w in zip(probs[selected].split(32768), wins[selected].split(32768)):
                mixture = (1 - grid) * p[:, None] + grid * w[:, None]
                sums += mixture.log().sum(0, dtype=torch.float64)
            weights[head] = grid[sums.argmax()]
    return weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="local Transformer checkpoint")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", DATA)))
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3, 4, 5, 7])
    parser.add_argument("--slots", type=int, default=2048)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--eval-context", type=int, default=4096)
    parser.add_argument("--eval-windows", type=int, default=96)
    parser.add_argument("--eval-full", action="store_true",
                        help="score every predictable byte in the validation and test splits")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--scaled-votes", action="store_true",
                        help="use smaller spike units for shorter suffixes")
    parser.add_argument("--spike-warmup", type=int, default=0,
                        help="number of readout updates with continuous strengths before quantizing")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.ctx, args.steps, args.bs, args.eval_context, args.eval_windows, args.threads) < 1 or args.lr <= 0:
        parser.error("context lengths, steps, batch sizes, threads and learning rate must be positive")
    if not 0 <= args.spike_warmup < args.steps:
        parser.error("spike warmup must be nonnegative and leave at least one quantized update")
    name = f"successor_s{args.seed}_{args.tag}"
    if result_exists("lm_successor", name, args.force):
        return
    # Experiment artifacts store JSON-safe CLI arguments and path metadata.
    args.base, args.data = str(args.base.expanduser()), str(args.data.expanduser())
    setup(args)
    try:
        device = resolve_device(args.device)
        output = checkpoint_path(args)
        train, val, test = load_splits(Path(args.data))
        checkpoint = torch.load(args.base, map_location="cpu", weights_only=True)
        if checkpoint["task"] != "lm" or checkpoint["cfg"]["attn"] != "transformer_local":
            raise ValueError("the base checkpoint must come from train_lm.py --variant transformer_local")
        stride = (len(val) - args.eval_context - 1) // args.eval_windows
        if (args.ctx >= len(train) - 1 or args.eval_context >= len(val)
                or (not args.eval_full and stride < args.eval_context)):
            raise ValueError("training or disjoint evaluation windows do not fit the corpus")
        cfg = replace(SNNConfig(**checkpoint["cfg"]), attn="transformer_successor",
                      successor_orders=tuple(args.orders), successor_slots=args.slots,
                      successor_scaled_votes=args.scaled_votes)
        kwargs = checkpoint["model_kwargs"]
        model = make_lm(cfg, **kwargs)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if set(missing) != {f"readout.{key}" for key in model.readout.state_dict()} or unexpected:
            raise ValueError("base checkpoint does not match the local Transformer backbone")
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    model.to(device).eval().requires_grad_(False)
    model.readout.requires_grad_(True)
    analog = deepcopy(model.readout)
    optimizer = torch.optim.AdamW([*model.readout.parameters(), *analog.parameters()], lr=args.lr)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for step in range(args.steps):
        tokens, targets = (chunk.to(device) for chunk in random_batch(train, args.bs, args.ctx, rng))
        with torch.no_grad():
            log_probs, successors, ages = model.evidence(tokens)
            base_prob = target_probability(log_probs, targets)
            correct = successors == targets[..., None]
        pulses = model.readout(log_probs, successors, ages, tokens, quantized=step >= args.spike_warmup)
        soft = analog(log_probs, successors, ages, tokens, quantized=False)
        spike_loss = next_byte_nll(base_prob, correct, pulses).mean()
        analog_loss = next_byte_nll(base_prob, correct, soft).mean()
        optimizer.zero_grad(set_to_none=True)
        (spike_loss + analog_loss).backward()
        optimizer.step()
        if (step + 1) % 50 == 0 or step + 1 == args.steps:
            print(f"step {step + 1}: train spike {spike_loss.item() / math.log(2):.4f}, "
                  f"analog {analog_loss.item() / math.log(2):.4f} bpb", flush=True)

    validation, validation_samples = measure(model, analog, val, args, device)
    lambdas = fit_fixed_cache(validation_samples, args.orders)
    validation["fixed_bpb"] = fixed_cache_bpb(validation_samples, lambdas)
    heldout, test_samples = measure(model, analog, test, args, device)
    heldout["fixed_bpb"] = fixed_cache_bpb(test_samples, lambdas)
    for split, score in (("val", validation), ("test", heldout)):
        print(f"{split}: {score}", flush=True)
    if output is not None:
        save_checkpoint(output, model, kwargs, args=args, task="lm", name=name)
    save_result("lm_successor", name, {
        "args": vars(args), "device": str(device), "data_sha1": ENWIK8_SHA1, "cfg": cfg.to_dict(),
        "base_run": checkpoint["run_name"], "base_params": count_params(model) - count_params(model.readout),
        "readout_params": count_params(model.readout), "calibration_bytes": args.steps * args.bs * args.ctx,
        "table_bytes_per_stream": len(args.orders) * args.slots * 3 * 8,
        "fixed_lambdas": dict(zip(args.orders, lambdas.tolist())), "val": validation, "test": heldout,
    }, overwrite=args.force)


if __name__ == "__main__":
    main()
