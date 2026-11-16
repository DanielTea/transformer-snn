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
from snnformer.event_lm import EventLM, EventLMConfig, _delay_group_widths
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
                layer.trace_read_phase = None
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


def probe_temporal_groups(model, window, chunk_size):
    groups = [{"delays": [], "axon_cells": 0, "held_cells": 0, "columns": []}
              for _ in model.blocks[0].widths]
    hidden = model.cfg.hidden
    for block in model.blocks:
        offset = 0
        for width, delay, group in zip(block.widths, block.local_delays, groups, strict=True):
            group["delays"].append(delay)
            group["axon_cells"] += delay * width
            group["held_cells"] += width
            for route in (2, 3):
                first = route * hidden + offset
                group["columns"].append(block.project.weight[:, first:first + width])
            offset += width

    intact = score(model, window, chunk_size)
    for group in groups:
        columns = group.pop("columns")
        group["muted_bpb"] = score_without_synapses(model, window, chunk_size, columns)
    return {"intact_bpb": intact, "muted_temporal_groups": groups}


def added_expansion_rows(block, previous, weight):
    before = _delay_group_widths(previous, len(block.widths)) if isinstance(previous, int) else tuple(previous)
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


@torch.inference_mode()
def probe_trace_input_gains(model, window, chunk_size, gains):
    prefix = window[:, :min(4096, window.shape[1] - 1)].cpu()
    weights = [block.trace.input.weight for block in model.blocks]
    originals = [weight.clone() for weight in weights]
    rows = []
    try:
        for gain in gains:
            for weight, original in zip(weights, originals, strict=True):
                weight.copy_(original * gain)
            loss = score(model, window, chunk_size)
            native = model.to_integer()
            emissions = []

            def count_pulses(_module, _inputs, pulses):
                emissions.append(pulses.abs().sum(dtype=torch.int64).item())

            handles = [block.trace.input.emit.register_forward_hook(count_pulses) for block in native.blocks]
            try:
                native(prefix)
            finally:
                for handle in handles:
                    handle.remove()
            rows.append({"gain_factor": gain, "validation_bpb": loss,
                         "native_input_pulses_per_byte": sum(emissions) / prefix.numel()})
    finally:
        for weight, original in zip(weights, originals, strict=True):
            weight.copy_(original)
    return {"pulse_sample_bytes": prefix.numel(), "rows": rows}


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
        thresholds = [block.inhibition_thresholds for block in model.blocks]
        hooks = [module.register_forward_hook(fix_count(module.quanta_per_threshold)) for module in thresholds]
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
    if model.cfg.trace_read_drive is not None:
        weights = [block.trace.read_drive.weight for block in model.blocks]
        result["zero_trace_read_drive_bpb"] = score_without_synapses(model, window, chunk_size, weights)
        other_source = "present" if model.cfg.trace_read_drive == "write" else "write"
        result["swapped_trace_read_drive_bpb"] = score_with_options(model, window, chunk_size,
                                                                  trace_read_drive=other_source)
    if model.cfg.trace_tap_read is not None:
        weights = [block.trace_taps.weight for block in model.blocks]
        result["zero_tap_correction_bpb"] = score_without_synapses(model, window, chunk_size, weights)
        if model.cfg.trace_tap_read == "race":
            result["fixed_tap_on_race_weights_bpb"] = score_with_options(model, window, chunk_size, trace_tap_read="fixed")
    if model.cfg.steering_transfer is not None:
        weights = [weight for block in model.blocks for weight in block.steering_transfer.weights]
        result["zero_steering_transfer_bpb"] = score_without_synapses(model, window, chunk_size, weights)
        source = model.cfg.steering_transfer
        other_source = "present" if source == "delayed" else "delayed"
        result[f"{other_source}_transfer_on_{source}_weights_bpb"] = score_with_options(
            model, window, chunk_size, steering_transfer=other_source,
        )
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
        charge_bits = EventLMConfig.router_charge_bits
        result["six_bit_router_bpb"] = score_with_options(
            model, window, chunk_size, router_charge_bits=charge_bits,
            inhibition_threshold_bits=min(model.cfg.inhibition_threshold_bits, charge_bits),
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
    intervention = parser.add_mutually_exclusive_group()
    intervention.add_argument("--tag-read-gains", type=float, nargs="+",
                              help="intervene on all read competition conductances and score the validation text")
    intervention.add_argument("--trace-input-gains", type=float, nargs="+",
                              help="scale the trained trace input synapses and score the validation text")
    intervention.add_argument("--temporal-groups", action="store_true",
                              help="mute the held-read and arriving bypass projections for one delay group at a time")
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
    if args.trace_input_gains is not None and (not model.cfg.trace_integration
            or any(not math.isfinite(gain) or gain < 0 for gain in args.trace_input_gains)):
        parser.error("trace input gain interventions require a trained trace input and finite nonnegative gains")
    if args.temporal_groups and not model.cfg.local_trace:
        parser.error("temporal group interventions require the held-charge router")
    _, validation, _ = load_splits(args.enwik8)
    if args.targets >= len(validation):
        parser.error("the requested targets exceed the validation split")
    sample = validation[:args.targets + 1]
    window = torch.from_numpy(sample.copy()).to(device=device, dtype=torch.long)[None]
    initialization = checkpoint.get("initialization") or {}
    if args.tag_read_gains is not None:
        metrics = {"tag_read_gain_sweep": probe_tag_read_gains(model, args.checkpoint, window,
                                                              args.chunk_size, args.tag_read_gains)}
    elif args.trace_input_gains is not None:
        metrics = {"trace_input_gain_sweep": probe_trace_input_gains(
            model, window, args.chunk_size, args.trace_input_gains,
        )}
    elif args.temporal_groups:
        metrics = probe_temporal_groups(model, window, args.chunk_size)
    else:
        previous_expansion = initialization.get("expansion_groups_before", initialization.get("expansion_width_before"))
        metrics = probe(model, window, args.chunk_size, residual_width_before=initialization.get("residual_width_before"),
                        expansion_width_before=previous_expansion,
                        depth_before=initialization.get("depth_before"))
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
