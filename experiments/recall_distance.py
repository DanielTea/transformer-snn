"""Associative recall with independently sampled distractor gaps.

Each record contains one key, distractors, and its value. Query tokens reuse
key IDs and occur after all records. Only query positions contribute to the
loss. Training varies the number of distractors per record and uses a fixed
context length. Evaluation reports accuracy at each exact gap, optionally
fixing both total context length and sampled binding/query identities across
gaps. In that mode, value positions can also be held fixed so that only
the key positions move.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

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
from snnformer import count_params


def make_batch(bs, n_keys, n_distractors, pairs, queries, min_gap, max_gap, context_length, gen,
               *, anchor_values=False):
    # Keys and values are distinct within each sequence. Distractor IDs are
    # disjoint from both sets, but the model sees only ordinary token IDs.
    key_ids, value_ids = torch.rand(bs, 2, n_keys, generator=gen).argsort(dim=-1).unbind(1)
    keys = key_ids[:, :pairs]
    values = value_ids[:, :pairs] + n_keys
    query_indices = torch.randint(pairs, (bs, queries), generator=gen)
    if min_gap == max_gap:
        gap = torch.full((bs, pairs), min_gap)
    else:
        gap = torch.randint(min_gap, max_gap + 1, (bs, pairs), generator=gen)
    if anchor_values:
        slot_width = (context_length - queries) // pairs
        if slot_width < max_gap + 2:
            raise ValueError("context is too short to place all values in fixed slots")
        slot_ends = torch.arange(1, pairs + 1) * slot_width - 1
        value_position = slot_ends.expand(bs, -1)
        key_position = value_position - gap - 1
    else:
        record_length = gap + 2
        key_position = record_length.cumsum(1) - record_length
        value_position = key_position + gap + 1

    tokens = torch.randint(2 * n_keys, 2 * n_keys + n_distractors, (bs, context_length), generator=gen)
    tokens.scatter_(1, key_position, keys)
    tokens.scatter_(1, value_position, values)
    tokens[:, -queries:] = keys.gather(1, query_indices)
    return tokens, values.gather(1, query_indices)


@torch.no_grad()
def evaluate(model, args, device, *, min_gap, max_gap, context_length, samples, seed, anchor_values=False):
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    correct = 0
    try:
        for start in range(0, samples, args.eval_bs):
            bs = min(args.eval_bs, samples - start)
            tokens, targets = make_batch(
                bs, args.keys, args.distractors, args.pairs, args.queries,
                min_gap, max_gap, context_length, gen, anchor_values=anchor_values,
            )
            predictions = model(tokens.to(device))[:, -args.queries:].argmax(-1)
            correct += (predictions == targets.to(device)).sum().item()
    finally:
        model.train(was_training)
    return correct / (samples * args.queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_sequence_args(parser)
    parser.set_defaults(causal_norm=True)
    parser.add_argument("--keys", type=int, default=32, help="size of each of the key and value vocabularies")
    parser.add_argument("--distractors", type=int, default=16, help="size of the separate distractor vocabulary")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--queries", type=int, default=6)
    parser.add_argument("--train-min-gap", type=int, default=0)
    parser.add_argument("--train-max-gap", type=int, default=4)
    parser.add_argument("--eval-gaps", type=int, nargs="+", default=[0, 4, 8, 16, 32])
    parser.add_argument("--eval-context-length", type=int,
                        help="use the same total evaluation context for every gap, including the query suffix")
    parser.add_argument("--anchor-eval-values", action="store_true",
                        help="hold value positions fixed across evaluation gaps (requires --eval-context-length)")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--bs", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-bs", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--eval-samples", type=int, default=1024)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--eval-seed", type=int, default=1234)
    args = parser.parse_args()
    if not 0 < args.pairs <= args.keys:
        parser.error("--pairs must be between 1 and --keys")
    if not 0 <= args.train_min_gap <= args.train_max_gap or min(args.eval_gaps) < 0:
        parser.error("gaps must be nonnegative and the training range must be ordered")
    positive = (
        args.queries, args.distractors, args.steps, args.bs, args.eval_every, args.eval_bs,
        args.val_samples, args.eval_samples, args.dim, args.heads, args.depth, args.threads,
    )
    if min(positive) < 1 or args.warmup < 0:
        parser.error("model, batch, and sample sizes must be positive, and warmup may be zero")
    if args.dim % args.heads:
        parser.error("--dim must be divisible by --heads")
    if len({args.seed, args.val_seed, args.eval_seed}) != 3:
        parser.error("training, validation, and held-out seeds must differ")
    if args.anchor_eval_values and args.eval_context_length is None:
        parser.error("--anchor-eval-values requires --eval-context-length")

    train_context = args.pairs * (args.train_max_gap + 2) + args.queries
    eval_contexts = {}
    for gap in sorted(set(args.eval_gaps)):
        minimum = args.pairs * (gap + 2) + args.queries
        length = args.eval_context_length if args.eval_context_length is not None else minimum
        if length < minimum:
            parser.error(f"evaluation gap {gap} requires context length at least {minimum}")
        eval_contexts[gap] = length
    cfg = build_cfg(args)
    rotary_positions = cfg.attn.startswith("transformer")
    if args.position_embeddings is None and not rotary_positions:
        cfg = replace(cfg, position_embeddings=False)
    ctx = max(train_context, *eval_contexts.values())
    if ctx > train_context and cfg.position_embeddings and not rotary_positions:
        parser.error("longer held-out sequences require --no-position to avoid untrained position embeddings")
    name = run_name(args, cfg) + f"_P{args.pairs}_G{args.train_min_gap}-{args.train_max_gap}"
    if args.anchor_eval_values:
        name += "_anchored"
    task = "recall_distance"
    if result_exists(task, name, args.force):
        return
    try:
        device = resolve_device(args.device)
        checkpoint = checkpoint_path(args)
    except ValueError as exc:
        parser.error(str(exc))
    setup(args)
    model_kwargs = dict(
        vocab=2 * args.keys + args.distractors, dim=args.dim, depth=args.depth, heads=args.heads, ctx=ctx,
    )
    model = make_lm(cfg, **model_kwargs).to(device)
    optimizer = make_optimizer(model, args.lr, args.wd)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    timer, curve = Timer(), []
    for step in range(args.steps):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step, args.steps, args.warmup, args.lr)
        tokens, targets = make_batch(
            args.bs, args.keys, args.distractors, args.pairs, args.queries,
            args.train_min_gap, args.train_max_gap, train_context, gen,
        )
        logits = model(tokens.to(device))[:, -args.queries:]
        loss = F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = evaluate(
                model, args, device, min_gap=args.train_min_gap, max_gap=args.train_max_gap,
                context_length=train_context, samples=args.val_samples, seed=args.val_seed,
            )
            curve.append({"step": step + 1, "loss": loss.item(), "acc": acc})
            print(f"{args.variant} step {step+1} loss {loss.item():.3f} acc {acc:.3f} ({timer():.0f}s)", flush=True)

    final_acc = evaluate(
        model, args, device, min_gap=args.train_min_gap, max_gap=args.train_max_gap,
        context_length=train_context, samples=args.eval_samples, seed=args.eval_seed,
    )
    final_by_gap = {
        gap: evaluate(
            model, args, device, min_gap=gap, max_gap=gap, context_length=length,
            samples=args.eval_samples, seed=args.eval_seed, anchor_values=args.anchor_eval_values,
        )
        for gap, length in eval_contexts.items()
    }
    params = count_params(model)
    if checkpoint is not None:
        save_checkpoint(checkpoint, model, model_kwargs, args=args, task=task, name=name)
    save_result(
        task, name,
        {
            "args": vars(args), "cfg": cfg.to_dict(), "device": str(device), "params": params,
            "train_context_length": train_context, "eval_context_length_by_gap": eval_contexts,
            "final_acc": final_acc, "final_acc_by_gap": final_by_gap, "curve": curve, "seconds": timer(),
        },
        overwrite=args.force,
    )
    scores = " ".join(f"G{gap}@{eval_contexts[gap]}={acc:.4f}" for gap, acc in final_by_gap.items())
    print(f"FINAL {name} train_range={final_acc:.4f} {scores} params={params}")


if __name__ == "__main__":
    main()
