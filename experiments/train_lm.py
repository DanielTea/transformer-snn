"""Byte-level language modelling on enwik8 (standard 90M/5M/5M split), reporting bits-per-byte.

Also records the validation learning curve against training tokens, which is
how sample efficiency is compared between variants.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    Timer,
    add_common_args,
    add_sequence_args,
    build_cfg,
    checkpoint_path,
    cosine_lr,
    make_lm,
    make_optimizer,
    resolve_device,
    result_exists,
    run_name,
    save_checkpoint,
    save_result,
    setup,
)
from snnformer import count_ops, count_params

DATA = Path("/home/user/data/enwik8")
ENWIK8_BYTES = 100_000_000
# Dataset publisher's checksum: https://www.mattmahoney.net/dc/textdata.html
ENWIK8_SHA1 = "57b8363b814821dc9d47aa4d41f58733519076b2"


def load_splits(path: Path = DATA):
    if path.stat().st_size != ENWIK8_BYTES:
        raise ValueError(f"{path}: expected a 100,000,000-byte raw enwik8 file; extract enwik8.zip first")
    raw = path.read_bytes()
    if hashlib.sha1(raw, usedforsecurity=False).hexdigest() != ENWIK8_SHA1:
        raise ValueError(f"{path}: checksum does not match canonical enwik8")
    data = np.frombuffer(raw, dtype=np.uint8)
    return data[:90_000_000], data[90_000_000:95_000_000], data[95_000_000:]


def random_batch(data, bs, ctx, gen):
    ix = torch.randint(0, len(data) - ctx - 1, (bs,), generator=gen).numpy()
    x = np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64)
    x = torch.from_numpy(x)
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def evaluate_bpb(model, data, ctx, n_batches, bs=32, offset=0, *, device="cpu"):
    """Mean bits-per-byte over evenly spaced windows with disjoint target bytes."""
    was_training = model.training
    model.eval()
    tot, cnt = 0.0, 0
    stride = (len(data) - offset - ctx - 1) // (n_batches * bs)
    for b in range(n_batches):
        ix = [offset + (b * bs + j) * stride for j in range(bs)]
        x = torch.from_numpy(np.stack([data[i : i + ctx + 1] for i in ix]).astype(np.int64)).to(device)
        logits = model(x[:, :-1])
        tot += F.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="sum").item()
        cnt += x[:, 1:].numel()
    model.train(was_training)
    return tot / cnt / math.log(2)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, bytes_only=True)
    add_sequence_args(p)
    p.add_argument("--data", default=os.environ.get("ENWIK8_PATH", str(DATA)),
                   help="path to the raw enwik8 file (also accepted in ENWIK8_PATH)")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-bs", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--final-eval-batches", type=int, default=150)
    args = p.parse_args()
    eval_sizes = (args.eval_every, args.eval_bs, args.eval_batches, args.final_eval_batches)
    if min(args.ctx, args.bs, args.steps, *eval_sizes) < 1 or args.warmup < 0:
        p.error("context, training and evaluation sizes must be positive; warmup may be zero")
    setup(args)
    cfg = build_cfg(args)
    name = run_name(args, cfg)
    if result_exists("lm", name, args.force):
        return
    try:
        device = resolve_device(args.device)
        checkpoint = checkpoint_path(args)
        train, val, test = load_splits(Path(args.data).expanduser())
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    if (len(val) - args.ctx - 1) // (max(args.eval_batches, args.final_eval_batches) * args.eval_bs) < args.ctx:
        p.error("evaluation windows would overlap; reduce --eval-bs or the number of evaluation batches")
    model_kwargs = dict(vocab=256, dim=args.dim, depth=args.depth, heads=args.heads, ctx=args.ctx)
    model = make_lm(cfg, **model_kwargs).to(device)
    opt = make_optimizer(model, args.lr, args.wd)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    timer, curve = Timer(), []
    tokens_per_step = args.bs * args.ctx
    print(f"{name}: params={count_params(model)}", flush=True)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.warmup, args.lr)
        x, y = random_batch(train, args.bs, args.ctx, gen)
        loss = F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            bpb = evaluate_bpb(model, val, args.ctx, args.eval_batches, bs=args.eval_bs, device=device)
            curve.append({"step": step + 1, "tokens": (step + 1) * tokens_per_step,
                          "train_bpb": loss.item() / math.log(2), "val_bpb": bpb})
            print(f"{name} step {step+1} train {loss.item()/math.log(2):.3f} val {bpb:.3f} bpb ({timer():.0f}s)",
                  flush=True)
    val_bpb = evaluate_bpb(model, val, args.ctx, args.final_eval_batches, bs=args.eval_bs, device=device)
    test_bpb = evaluate_bpb(model, test, args.ctx, args.final_eval_batches, bs=args.eval_bs, device=device)
    model.eval()
    energy = None
    if cfg.attn not in ("transformer_event", "transformer_successor"):
        with torch.no_grad(), count_ops(model, cfg.spike_levels, cfg.spike_unit) as c:
            x, _ = random_batch(val, 16, args.ctx, torch.Generator(device="cpu").manual_seed(0))
            model(x.to(device))
            c.samples += x.numel()  # per token
        energy = c.summary()
    if checkpoint is not None:
        save_checkpoint(checkpoint, model, model_kwargs, args=args, task="lm", name=name)
    save_result("lm", name, {"args": vars(args), "cfg": cfg.to_dict(), "device": str(device),
                             "params": count_params(model), "data_sha1": ENWIK8_SHA1,
                             "val_bpb": val_bpb, "test_bpb": test_bpb, "curve": curve,
                             "energy_per_token": energy, "seconds": timer()}, overwrite=args.force)
    print(f"FINAL {name} val_bpb={val_bpb:.4f} test_bpb={test_bpb:.4f} energy/token={energy}")


if __name__ == "__main__":
    main()
