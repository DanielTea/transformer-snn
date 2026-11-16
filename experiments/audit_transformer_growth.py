"""Check warm-start logits and gradients when widening a byte Transformer's SwiGLU layers."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from common import _atomic_output, make_lm, resolve_device
from train_event_lm import initialize_from_checkpoint, load_checkpoint
from train_lm import load_splits, random_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--enwik8", required=True, type=Path)
    parser.add_argument("--mlp-ratio", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="mps", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    parent, checkpoint = load_checkpoint(args.parent, torch.device("cpu"))
    if checkpoint["cfg"].get("attn") != "transformer":
        parser.error("use a full-attention byte language model checkpoint")
    options = checkpoint["model_kwargs"]
    widened = make_lm(replace(parent.cfg, mlp_ratio=args.mlp_ratio), **options)
    initialize_from_checkpoint(widened, options, args.parent, freeze_router=False, grow_feed_forward=True)
    old_width = parent.blocks[0].mlp_out.in_features
    new_width = widened.blocks[0].mlp_out.in_features

    training, validation, _ = load_splits(args.enwik8)
    sample = validation[:4096]
    tokens = torch.tensor(sample, dtype=torch.long, device=device).view(8, 512)
    parent = parent.to(device)
    widened = widened.to(device).eval()
    with torch.no_grad():
        before = parent(tokens)
        after = widened(tokens)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        max_difference = (before - after).abs().max().item()

    generator = torch.Generator().manual_seed(args.seed)
    inputs, labels = random_batch(training, 1, 512, generator)
    logits = widened.train()(inputs.to(device))
    F.cross_entropy(logits.flatten(0, 1), labels.to(device).flatten()).backward()
    gradients = [block.mlp_out.weight.grad[:, old_width:].count_nonzero().item() for block in widened.blocks]
    if not all(gradients):
        raise AssertionError("the added feed-forward units received no gradient in a block")

    with args.parent.open("rb") as parent_file:
        parent_digest = hashlib.file_digest(parent_file, "sha256").hexdigest()
    record = {
        "checkpoint": str(args.parent), "checkpoint_sha256": parent_digest,
        "validation_sha256": hashlib.sha256(sample).hexdigest(), "validation_bytes": sample.size,
        "seed": args.seed, "device": str(device), "gradient_training_targets": labels.numel(),
        "parent_parameters": sum(weight.numel() for weight in parent.parameters()),
        "grown_parameters": sum(weight.numel() for weight in widened.parameters()),
        "validation_logits_exact": True, "validation_logits_max_abs_difference": max_difference,
        "feed_forward_width_before": old_width, "feed_forward_width_after": new_width,
        "active_new_output_gradients_on_512_train_targets": gradients,
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
