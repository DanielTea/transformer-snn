"""Intervene on inhibition and local memory in a trained spike decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLM
from train_event_lm import scoring_logits
from train_lm import load_splits


@torch.inference_mode()
def score(model, window, chunk_size, *, reset_trace=False):
    state = None
    losses = []
    for offset in range(0, window.shape[1] - 1, chunk_size):
        section = window[:, offset:offset + chunk_size + 1]
        if reset_trace and state is not None:
            for layer in state.local_layers:
                layer.trace_charge = None
        counts, state = model.forward_chunk(section[:, :-1], state)
        logits = scoring_logits(counts, True, model.cfg.resolution)
        losses.append(F.cross_entropy(logits.flatten(0, 1), section[:, 1:].flatten(), reduction="sum").item())
    targets = window.numel() - window.shape[0]
    return math.fsum(losses) / (targets * math.log(2))


def watch(series):
    def observe(_module, _args, pulses):
        series.append(pulses.detach().cpu().to(torch.int32))
    return observe


def fix_count(count):
    def replace(_module, _args, pulses):
        return torch.full_like(pulses, count)
    return replace


@torch.inference_mode()
def score_without_synapses(model, window, chunk_size, weights):
    originals = [weight.clone() for weight in weights]
    try:
        for weight in weights:
            weight.zero_()
        return score(model, window, chunk_size)
    finally:
        for weight, original in zip(weights, originals, strict=True):
            weight.copy_(original)


def probe(model, window, chunk_size):
    pools = [block.activity_inhibition for block in model.blocks if block.activity_inhibition is not None]
    series = [[] for _ in pools]
    hooks = [pool.register_forward_hook(watch(counts)) for pool, counts in zip(pools, series, strict=True)]
    try:
        result = {"intact_bpb": score(model, window, chunk_size)}
    finally:
        for hook in hooks:
            hook.remove()
    if pools:
        targets = window.shape[1] - 1
        spans = [min(chunk_size, targets - offset) for offset in range(0, targets, chunk_size)]
        # Prefix replay supplies context to some decoders. Count only new bytes.
        medians = [int(torch.cat([count[:, -span:].flatten()
                                 for count, span in zip(counts, spans, strict=True)]).median())
                   for counts in series]
        hooks = [pool.register_forward_hook(fix_count(count)) for pool, count in zip(pools, medians, strict=True)]
        try:
            result["fixed_inhibition_bpb"] = score(model, window, chunk_size)
        finally:
            for hook in hooks:
                hook.remove()
        result["fixed_inhibition_pulses_by_block"] = medians
    if model.cfg.local_trace:
        result["periodic_trace_reset_bpb"] = score(model, window, chunk_size, reset_trace=True)
    if model.cfg.delay_filter:
        weights = [kernel.weight for block in model.blocks for kernel in block.delay_filters if kernel is not None]
        result["zero_delay_filter_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.opponent_steering:
        weights = [block.opponent_steering.weight for block in model.blocks]
        result["zero_opponent_steering_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.trace_read_veto:
        weights = [block.trace.read_veto.gain for block in model.blocks]
        result["zero_trace_read_veto_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--targets", type=int, default=32_768)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.targets, args.chunk_size) < 1:
        parser.error("targets and chunk size must be positive")
    if args.json.exists():
        parser.error(f"output already exists: {args.json}")
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, EventLM) or not (model.has_correction or model.cfg.local_trace):
        parser.error("a chunk-capable spiking checkpoint is required")
    if not (model.cfg.activity_inhibition or model.cfg.local_trace or model.cfg.delay_filter):
        parser.error("the checkpoint needs inhibition, a local trace, or delay filters")
    _, validation, _ = load_splits(args.enwik8)
    if args.targets >= len(validation):
        parser.error("the requested targets exceed the validation split")
    sample = validation[:args.targets + 1]
    window = torch.from_numpy(sample.copy()).to(device=device, dtype=torch.long)[None]
    record = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": "enwik8_validation",
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "target_bytes": args.targets,
        "chunk_bytes": args.chunk_size,
        **probe(model, window, args.chunk_size),
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
