"""Replay the feed-forward growth and zero-current steering checks on enwik8."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_spike_code import _IntegerExecution
from common import _atomic_output, make_lm, resolve_device
from probe_event_state import added_expansion_rows
from snnformer import count_params
from snnformer.event_lm import EventLM
from snnformer.transformer_reference import TransformerLM
from train_event_lm import initialize_from_checkpoint, load_checkpoint, scoring_logits
from train_lm import load_splits, random_batch


def audit_transformer(args, parent, checkpoint, validation, inputs, labels, device):
    if not isinstance(parent, TransformerLM) or parent.cfg.attn != "transformer":
        raise ValueError("feed-forward growth needs a full-attention Transformer checkpoint")
    cfg = replace(parent.cfg, mlp_ratio=args.mlp_ratio)
    kwargs = checkpoint["model_kwargs"]
    grown = make_lm(cfg, **kwargs)
    initialized = initialize_from_checkpoint(grown, kwargs, args.checkpoint, freeze_router=False, grow_feed_forward=True)
    previous_width = initialized["feed_forward_width_before"]
    grown.to(device).eval()
    windows = validation.reshape(8, 512).to(device)
    with torch.no_grad():
        before = parent.to(device)(windows)
        after = grown(windows)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    logits = grown.train()(inputs)
    F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
    gradients = [int(block.mlp_out.weight.grad[:, previous_width:].count_nonzero()) for block in grown.blocks]
    if not all(gradients):
        raise AssertionError("added Transformer units received no gradient in a block")
    return {
        "parent_parameters": count_params(parent), "grown_parameters": count_params(grown),
        "feed_forward_width_before": previous_width, "feed_forward_width_after": grown.blocks[0].mlp_out.in_features,
        "validation_logits_exact": True, "active_new_output_gradients_on_512_train_targets": gradients,
    }


def audit_steering(parent, checkpoint, validation, inputs, labels, device):
    prior_width = (checkpoint.get("initialization") or {}).get("expansion_width_before")
    if not isinstance(parent, EventLM) or prior_width is None or parent.cfg.steering_gradient_at_zero:
        raise ValueError("the steering check needs an expanded SNN trained with the ordinary rectifier")
    candidate = EventLM(replace(parent.cfg, steering_gradient_at_zero=True)).eval()
    candidate.load_state_dict(checkpoint["state_dict"])
    native = [model.to_integer() for model in (parent, candidate)]
    with torch.no_grad(), _IntegerExecution():
        torch.testing.assert_close(native[0](validation), native[1](validation), rtol=0, atol=0)
    result = {}
    for label, model in (("ordinary", parent), ("right_derivative", candidate)):
        counts = model.to(device).train()(inputs)
        logits = scoring_logits(counts, True, model.cfg.resolution)
        F.cross_entropy(logits.flatten(0, 1), labels.flatten()).backward()
        gradients = []
        for block in model.blocks:
            added_rows = added_expansion_rows(block, prior_width, block.expand.weight.grad)
            steering = added_rows[len(block.widths):]
            gradients.append(sum(int(row.count_nonzero()) for row in steering))
        result[label] = gradients
    if not all(result["right_derivative"]):
        raise AssertionError("added steering neurons received no gradient in a block")
    return {"native_counts_equal": True, "new_steering_gradients_by_block": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("transformer", "steering"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--mlp-ratio", type=float, help="widen the Transformer's SwiGLU to this ratio")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if (args.study == "transformer") != (args.mlp_ratio is not None):
        parser.error("use --mlp-ratio to specify the new width for the transformer check")

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    train, validation, _ = load_splits(args.enwik8)
    sample = validation[:4096]
    tokens = torch.tensor(sample, dtype=torch.long).unsqueeze(0)
    generator = torch.Generator().manual_seed(args.seed)
    inputs, labels = (batch.to(device) for batch in random_batch(train, 1, 512, generator))
    parent, checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if args.study == "transformer":
        measured = audit_transformer(args, parent, checkpoint, tokens, inputs, labels, device)
    else:
        measured = audit_steering(parent, checkpoint, tokens, inputs, labels, device)
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest,
        "validation_sha256": hashlib.sha256(sample).hexdigest(), "validation_bytes": sample.size,
        "seed": args.seed, "device": str(device), "gradient_training_targets": labels.numel(), **measured,
    }
    with _atomic_output(args.json, "w", overwrite=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
