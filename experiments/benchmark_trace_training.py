"""Compare charge gradient rules on the same enwik8 training batch."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
from torch.nn import functional as F

from common import _atomic_output
from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLM
from train_event_lm import scoring_logits
from train_lm import load_splits


def train_time(model, inputs, labels):
    model.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    start = perf_counter()
    counts = model(inputs)
    logits = scoring_logits(counts, True, model.cfg.resolution)
    F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
    torch.mps.synchronize()
    return perf_counter() - start, counts.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--site", choices=("trace", "bank"), default="trace")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.context, args.batch_size, args.trials) < 1:
        parser.error("batch size, context, and trials must be positive")
    torch.set_num_threads(4)
    parent, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if not isinstance(parent, EventLM) or not (parent.cfg.local_trace if args.site == "trace" else parent.has_correction):
        parser.error(f"the checkpoint needs a {args.site} charge recurrence")
    device = torch.device("mps")
    variants = {}
    for name, credit in (("original", False), ("write_credit", True)):
        model = EventLM(replace(parent.cfg, **{f"{args.site}_write_credit": credit}))
        model.load_state_dict(parent.state_dict())
        variants[name] = model.to(device).train()

    training, _, _ = load_splits(args.enwik8)
    sample = training[:args.batch_size * (args.context + 1)]
    if sample.size != args.batch_size * (args.context + 1):
        parser.error("the requested windows exceed the training split")
    windows = torch.from_numpy(sample.copy()).reshape(args.batch_size, args.context + 1).to(device=device, dtype=torch.long)
    inputs, labels = windows[:, :-1], windows[:, 1:]
    timings = {name: [] for name in variants}
    for trial in range(args.trials + 1):
        scores = []
        order = list(variants) if trial % 2 else list(reversed(variants))
        for name in order:
            elapsed, counts = train_time(variants[name], inputs, labels)
            scores.append(counts)
            if trial:
                timings[name].append(elapsed)
        torch.testing.assert_close(*scores, rtol=0, atol=0)
    seconds = {name: median(records) for name, records in timings.items()}
    record = {
        "device": str(device),
        "split": "enwik8_train",
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "checkpoint": str(args.checkpoint),
        "site": args.site,
        "batch_size": args.batch_size,
        "context_bytes": args.context,
        "trials_after_warmup": args.trials,
        "identical_training_forward_vocabulary_counts": True,
        "language_loss_forward_backward_seconds": seconds,
        "speedup": seconds["original"] / seconds["write_credit"],
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
