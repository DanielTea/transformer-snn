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
    TRANSFORMER_MIXERS,
    Timer,
    add_common_args,
    add_sequence_args,
    build_cfg,
    checkpoint_path,
    cosine_lr,
    make_optimizer,
    make_lm,
    resolve_device,
    result_exists,
    run_name,
    save_checkpoint,
    save_result,
    setup,
)
from snnformer import count_params


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
def evaluate(model, args, device, *, pairs, samples, seed):
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    correct = 0
    for start in range(0, samples, args.eval_bs):
        bs = min(args.eval_bs, samples - start)
        x, y = make_batch(bs, args.keys, pairs, args.queries, gen)
        pred = model(x.to(device))[:, -args.queries :].argmax(-1)
        correct += (pred == y[:, -args.queries :].to(device)).sum().item()
    model.train(was_training)
    return correct / (samples * args.queries)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p)
    add_sequence_args(p)
    p.add_argument("--keys", type=int, default=64)
    p.add_argument("--pairs", type=int, default=16)
    p.add_argument("--queries", type=int, default=16)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-bs", type=int, default=256)
    p.add_argument("--val-samples", type=int, default=1024)
    p.add_argument("--eval-samples", type=int, default=4096)
    p.add_argument("--val-seed", type=int, default=12345)
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--eval-pairs", type=int, nargs="+", default=[],
                   help="additional held-out pair counts, keeping the same number of queries")
    args = p.parse_args()
    if not all(0 < pairs <= args.keys for pairs in (args.pairs, *args.eval_pairs)):
        p.error("each pair count must be between 1 and --keys")
    positive = (args.queries, args.steps, args.bs, args.eval_every, args.eval_bs,
                args.val_samples, args.eval_samples)
    if min(positive) < 1 or args.warmup < 0:
        p.error("query count, steps, batch sizes and evaluation sizes must be positive; warmup may be zero")
    if len({args.seed, args.val_seed, args.eval_seed}) != 3:
        p.error("training, validation and held-out evaluation seeds must differ")
    setup(args)
    cfg = build_cfg(args)
    pair_counts = sorted({args.pairs, *args.eval_pairs})
    max_pairs = pair_counts[-1]
    if max_pairs > args.pairs and cfg.position_embeddings and cfg.attn not in TRANSFORMER_MIXERS:
        p.error("longer held-out sequences require --no-position to avoid untrained position embeddings")
    name = run_name(args, cfg) + f"_P{args.pairs}"
    if result_exists("recall", name, args.force):
        return
    try:
        device = resolve_device(args.device)
        checkpoint = checkpoint_path(args)
    except ValueError as exc:
        p.error(str(exc))
    ctx = 2 * max_pairs + args.queries
    model_kwargs = dict(vocab=2 * args.keys, dim=args.dim, depth=args.depth, heads=args.heads, ctx=ctx)
    model = make_lm(cfg, **model_kwargs).to(device)
    opt = make_optimizer(model, args.lr, args.wd)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    timer, curve = Timer(), []
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step, args.steps, args.warmup, args.lr)
        x, y = make_batch(args.bs, args.keys, args.pairs, args.queries, gen)
        loss = F.cross_entropy(model(x.to(device)).flatten(0, 1), y.to(device).flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = evaluate(model, args, device, pairs=args.pairs, samples=args.val_samples, seed=args.val_seed)
            curve.append({"step": step + 1, "loss": loss.item(), "acc": acc})
            print(f"{args.variant} step {step+1} loss {loss.item():.3f} acc {acc:.3f} ({timer():.0f}s)", flush=True)
    final_by_pairs = {
        pairs: evaluate(model, args, device, pairs=pairs, samples=args.eval_samples, seed=args.eval_seed)
        for pairs in pair_counts
    }
    final = final_by_pairs[args.pairs]
    params = count_params(model)
    if checkpoint is not None:
        save_checkpoint(checkpoint, model, model_kwargs, args=args, task="recall", name=name)
    save_result(
        "recall",
        name,
        {"args": vars(args), "cfg": cfg.to_dict(), "device": str(device), "params": params,
         "final_acc": final, "final_acc_by_pairs": final_by_pairs, "curve": curve, "seconds": timer()},
        overwrite=args.force,
    )
    scores = " ".join(f"P{pairs}={acc:.4f}" for pairs, acc in final_by_pairs.items())
    print(f"FINAL {name} {scores} params={params}")


if __name__ == "__main__":
    main()
