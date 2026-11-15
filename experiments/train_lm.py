"""Byte-level language modelling on enwik8 (standard 90M/5M/5M split), reporting bits-per-byte.

Also records the validation learning curve against training tokens, which is
how sample efficiency is compared between variants.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    Timer,
    add_common_args,
    build_cfg,
    cosine_lr,
    make_optimizer,
    run_name,
    save_result,
    setup,
)
from snnformer import SpikingLM, count_ops, count_params

DATA = Path("/home/user/data/enwik8")


def load_splits():
    raw = np.frombuffer(DATA.read_bytes(), dtype=np.uint8)
    return raw[:90_000_000], raw[90_000_000:95_000_000], raw[95_000_000:]


def random_batch(data, bs, ctx, gen):
    ix = torch.randint(0, len(data) - ctx - 1, (bs,), generator=gen).numpy()
    x = np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x)
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate_bpb(model, data, ctx, n_batches, bs=32, offset=0):
    """Mean bits-per-byte over contiguous, non-overlapping windows (deterministic)."""
    model.eval()
    tot, cnt = 0.0, 0
    stride = (len(data) - ctx - 1) // (n_batches * bs)
    for b in range(n_batches):
        ix = [offset + (b * bs + j) * stride for j in range(bs)]
        x = torch.from_numpy(np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64))
        logits = model(x[:, :-1])
        tot += F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train()
    return tot / cnt / math.log(2)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--final-eval-batches", type=int, default=150)
    args = p.parse_args()
    setup(args)
    cfg = build_cfg(args)
    train, val, test = load_splits()
    model = SpikingLM(cfg, vocab=256, dim=args.dim, depth=args.depth, heads=args.heads, ctx=args.ctx)
    opt = make_optimizer(model, args.lr, args.wd)
    gen = torch.Generator().manual_seed(args.seed)
    timer, curve = Timer(), []
    tokens_per_step = args.bs * args.ctx
    name = run_name(args, cfg)
    print(f"{name}: params={count_params(model)}", flush=True)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, 200, args.lr)
        x, y = random_batch(train, args.bs, args.ctx, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            bpb = evaluate_bpb(model, val, args.ctx, args.eval_batches)
            curve.append({"step": step + 1, "tokens": (step + 1) * tokens_per_step,
                          "train_bpb": loss.item() / math.log(2), "val_bpb": bpb})
            print(f"{name} step {step+1} train {loss.item()/math.log(2):.3f} val {bpb:.3f} bpb ({timer():.0f}s)",
                  flush=True)
    val_bpb = evaluate_bpb(model, val, args.ctx, args.final_eval_batches)
    test_bpb = evaluate_bpb(model, test, args.ctx, args.final_eval_batches)
    model.eval()
    with torch.no_grad(), count_ops(model, cfg.spike_levels, cfg.spike_unit) as c:
        x, _ = random_batch(val, 16, args.ctx, torch.Generator().manual_seed(0))
        model(x)
        c.samples += x.numel()  # per token
    energy = c.summary()
    save_result("lm", name, {"args": vars(args), "cfg": cfg.to_dict(), "params": count_params(model),
                             "val_bpb": val_bpb, "test_bpb": test_bpb, "curve": curve,
                             "energy_per_token": energy, "seconds": timer()})
    print(f"FINAL {name} val_bpb={val_bpb:.4f} test_bpb={test_bpb:.4f} energy/token={energy}")


if __name__ == "__main__":
    main()
