"""Intervene on inhibition and local memory in a trained spike decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLM, EventLMConfig, _delay_group_widths, _PACKET_UNITS
from train_event_lm import initialize_from_checkpoint, scoring_logits
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


@torch.inference_mode()
def score_with_options(model, window, chunk_size, **options):
    variant = EventLM(replace(model.cfg, **options)).eval().to(window.device)
    variant.load_state_dict(model.state_dict())
    return score(variant, window, chunk_size)


def added_expansion_rows(block, previous_width, weight):
    before = _delay_group_widths(previous_width, len(block.widths))
    return [group[width:] for group, width in zip(weight.split(block.widths * 2), before * 2, strict=True)]


@torch.inference_mode()
def probe_tag_read_gains(model, checkpoint_path, window, chunk_size, gains):
    if not model.cfg.tag_read_competition:
        model = EventLM(replace(model.cfg, tag_read_competition=True))
        initialize_from_checkpoint(model, {}, checkpoint_path, freeze_router=False)
        model.to(window.device).eval()
    metrics = []
    prefix = window[:, :min(4096, window.shape[1] - 1)].cpu()
    gain = model.prediction_memory.read_inhibition.gain
    for value in gains:
        gain.fill_(value)
        native = model.to_integer()
        stats = {}
        native.forward_chunk(prefix, stats=stats)
        metrics.append({
            "gain_ticks": native.prediction_memory.read_inhibition.gains().tolist(),
            "validation_bpb": score(model, window, chunk_size),
            "native_activity_bytes": prefix.numel(),
            "memory_stats": {name: int(count) for name, count in stats.items()},
        })
    return metrics


def probe(model, window, chunk_size, *, residual_width_before=None, expansion_width_before=None, depth_before=None):
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
    if model.cfg.redistribute_inhibition:
        weights = [block.inhibition_redistribution.synapse.weight for block in model.blocks]
        result["zero_inhibition_transfer_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.learn_inhibition_thresholds:
        hooks = [block.inhibition_thresholds.register_forward_hook(fix_count(_PACKET_UNITS)) for block in model.blocks]
        try:
            result["fixed_group_thresholds_bpb"] = score(model, window, chunk_size)
        finally:
            for hook in hooks:
                hook.remove()
    if model.cfg.local_trace:
        result["periodic_trace_reset_bpb"] = score(model, window, chunk_size, reset_trace=True)
    if model.cfg.trace_integration:
        weights = [block.trace.input.weight for block in model.blocks]
        result["zero_trace_input_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.prediction_memory is not None:
        result["mute_bank_corrections_bpb"] = score_without_synapses(
            model, window, chunk_size, [model.recall_synapse.weight],
        )
    if model.cfg.tag_read_competition:
        gain = model.prediction_memory.read_inhibition.gain
        result["zero_tag_read_competition_bpb"] = score_without_synapses(model, window, chunk_size, [gain])
    if model.cfg.late_steering:
        weights = [block.late_steering.gain for block in model.blocks]
        result["zero_late_steering_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.router_charge_bits > EventLMConfig.router_charge_bits:
        result["six_bit_router_bpb"] = score_with_options(
            model, window, chunk_size, router_charge_bits=EventLMConfig.router_charge_bits,
        )
    if model.cfg.delay_filter:
        weights = [kernel.weight for block in model.blocks for kernel in block.delay_filters if kernel is not None]
        result["zero_delay_filter_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.opponent_steering:
        weights = [block.opponent_steering.weight for block in model.blocks]
        result["zero_opponent_steering_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.trace_read_veto:
        weights = [block.trace.read_veto.gain for block in model.blocks]
        result["zero_trace_read_veto_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.trace_write_confirm:
        weights = [block.trace.write_confirm.gain for block in model.blocks]
        result["zero_trace_write_confirm_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.spread_coincidence:
        weights = [block.spread_coincidence.mix for block in model.blocks]
        result["zero_spread_coincidence_bpb"] = score_without_synapses(model, window, chunk_size, weights)
    if model.cfg.residual_membrane:
        result["reset_residual_membrane_bpb"] = score_with_options(model, window, chunk_size, residual_membrane=False)
    if model.cfg.tag_phase_address:
        result["one_phase_address_bpb"] = score_with_options(model, window, chunk_size, tag_phase_address=False)
    if model.cfg.trace_coincidence_start is not None:
        result["ordinary_coincidence_bpb"] = score_with_options(model, window, chunk_size, trace_coincidence_start=None)
    if residual_width_before is not None:
        output_synapses = [model.input, *model.delayed_input, *(block.project for block in model.blocks)]
        output_synapses += [block.memory_residual for block in model.blocks if block.memory_residual is not None]
        if model.prediction_memory is not None:
            output_synapses.append(model.recall_synapse)
        extra_outputs = [synapse.weight[:, residual_width_before:] if synapse.byte_input
                         else synapse.weight[residual_width_before:] for synapse in output_synapses]
        result["silence_added_residual_bpb"] = score_without_synapses(model, window, chunk_size, extra_outputs)
    if expansion_width_before is not None:
        new_rows = [rows for block in model.blocks
                    for rows in added_expansion_rows(block, expansion_width_before, block.expand.weight)]
        result["silence_added_expansion_bpb"] = score_without_synapses(model, window, chunk_size, new_rows)
    if depth_before is not None:
        added_projections = [block.project.weight for block in model.blocks[depth_before:]]
        result["silence_added_blocks_bpb"] = score_without_synapses(model, window, chunk_size, added_projections)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--targets", type=int, default=32_768)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--tag-read-gains", type=float, nargs="+",
                        help="intervene on all read competition conductances and score the validation text")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.targets, args.chunk_size) < 1:
        parser.error("targets and chunk size must be positive")
    if args.json.exists():
        parser.error(f"output already exists: {args.json}")
    torch.set_num_threads(4)
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if not isinstance(model, EventLM) or not (model.has_correction or model.cfg.local_trace):
        parser.error("a chunk-capable spiking checkpoint is required")
    if not (model.cfg.activity_inhibition or model.cfg.local_trace or model.cfg.delay_filter):
        parser.error("the checkpoint needs inhibition, a local trace, or delay filters")
    if args.tag_read_gains is not None and (not model.cfg.match_forecast_key
            or any(not 0 <= gain <= 1 for gain in args.tag_read_gains)):
        parser.error("tag read gain interventions require forecast tags and gains in [0, 1]")
    _, validation, _ = load_splits(args.enwik8)
    if args.targets >= len(validation):
        parser.error("the requested targets exceed the validation split")
    sample = validation[:args.targets + 1]
    window = torch.from_numpy(sample.copy()).to(device=device, dtype=torch.long)[None]
    initialization = checkpoint.get("initialization") or {}
    if args.tag_read_gains is None:
        metrics = probe(model, window, args.chunk_size, residual_width_before=initialization.get("residual_width_before"),
                        expansion_width_before=initialization.get("expansion_width_before"),
                        depth_before=initialization.get("depth_before"))
    else:
        metrics = {"tag_read_gain_sweep": probe_tag_read_gains(model, args.checkpoint, window,
                                                              args.chunk_size, args.tag_read_gains)}
    record = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": "enwik8_validation",
        "sample_sha256": hashlib.sha256(sample).hexdigest(),
        "target_bytes": args.targets,
        "chunk_bytes": args.chunk_size,
        **metrics,
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
