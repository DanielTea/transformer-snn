"""Multi-query associative recall (MQAR, Arora et al. 2023 "Zoology").

A sequence of P key-value pairs is followed by Q queried keys; at every query
position the model must emit the value bound to that key.  This isolates the
ability of an attention mechanism to *select* one token among many -- the
operation softmax does well and softmax-free spiking attention is suspected to
do badly.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from common import (
    RESULTS,
    Timer,
    add_common_args,
    build_cfg,
    cosine_lr,
    make_optimizer,
    run_name,
    save_result,
    setup,
)
from snnformer import SpikingLM, count_params


def make_batch(bs: int, n_keys: int, pairs: int, queries: int, gen: torch.Generator):
    keys = torch.argsort(torch.rand(bs, n_keys, generator=gen), dim=1)[:, :pairs]
    vals = torch.randint(n_keys, 2 * n_keys, (bs, pairs), generator=gen)
    kv = torch.stack([keys, vals], dim=2).flatten(1)  # k1 v1 k2 v2 ...
    qi = torch.randint(0, pairs, (bs, queries), generator=gen)
    qk = keys.gather(1, qi)
    x = torch.cat([kv, qk], dim=1)
    y = torch.full_like(x, -100)
    y[:, 2 * pairs :] = vals.gather(1, qi)
    return x, y


@torch.no_grad()
def evaluate(model, args, n=2048):
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    correct = total = 0
    for _ in range(n // 256):
        x, y = make_batch(256, args.keys, args.pairs, args.queries, gen)
        pred = model(x).argmax(-1)
        m = y != -100
        correct += (pred[m] == y[m]).sum().item()
        total += m.sum().item()
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    p.add_argument("--keys", type=int, default=64)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--queries", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--force", action="store_true", help="re-run even if the result file exists")
    args = p.parse_args()
    setup(args)
    cfg = build_cfg(args)
    out = RESULTS / "recall" / (run_name(args, cfg) + f"_P{args.pairs}.json")
    if out.exists() and not args.force:
        print(f"skip {out.name} (exists)")
        return
    ctx = 2 * args.pairs + args.queries
    model = SpikingLM(cfg, vocab=2 * args.keys, dim=args.dim, depth=args.depth, heads=args.heads, ctx=ctx)
    opt = make_optimizer(model, args.lr, args.wd)
    gen = torch.Generator().manual_seed(args.seed)
    timer, curve = Timer(), []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, 100, args.lr)
        x, y = make_batch(args.bs, args.keys, args.pairs, args.queries, gen)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 250 == 0 or step + 1 == args.steps:
            acc = evaluate(model, args, n=1024)
            curve.append({"step": step + 1, "loss": loss.item(), "acc": acc})
            print(f"{args.variant} step {step+1} loss {loss.item():.3f} acc {acc:.3f} ({timer():.0f}s)", flush=True)
    final = evaluate(model, args, n=4096)
    name = run_name(args, cfg) + f"_P{args.pairs}"
    save_result(
        "recall",
        name,
        {"args": vars(args), "cfg": cfg.to_dict(), "params": count_params(model), "final_acc": final,
         "curve": curve, "seconds": timer()},
    )
    print(f"FINAL {name} acc={final:.4f} params={count_params(model)}")


if __name__ == "__main__":
    main()
