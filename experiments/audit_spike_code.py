"""Compare spike emissions and native inference for a router and its prediction banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from benchmark_decode import state_bytes
from common import _atomic_output, resolve_device
from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLM, EventLMConfig, _delay_group_widths, _IntegrateFire, _PACKET_TICKS
from train_lm import ENWIK8_SHA1, load_splits


class _IntegerExecution(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for item in tree_leaves((args, kwargs, result)):
            if isinstance(item, torch.Tensor) and (item.is_floating_point() or item.is_complex()):
                raise AssertionError(f"native decoding used {item.dtype} in {func}")
        self.operations += 1
        return result


def _forecast_errors(model, tokens, frequencies):
    _, predicted, observed = model(tokens, return_prediction=True)
    byte_ids = torch.arange(256)[None]
    codes = model.input_emit(model.input(byte_ids) * model.cfg.resolution)
    codes = model.prediction_memory.observation_code(codes)[0]
    if model.cfg.learn_context_keys:
        channels = model.cfg.forecast_key_channels
        codes = codes[..., channels:]
        predicted = predicted[..., channels:]
        observed = observed[..., channels:]
    sorted_codes, ids = codes.sort(dim=0)
    weights = frequencies[ids].cumsum(dim=0)
    index = (2 * weights >= frequencies.sum()).to(torch.int32).argmax(dim=0, keepdim=True)
    median = sorted_codes.gather(0, index)
    targets = observed[:, 1:]
    return {
        "forecast_absolute_error_counts": (predicted[:, :-1] - targets).abs().float().mean().item(),
        "train_unconditional_median_error_counts": (median - targets).abs().float().mean().item(),
    }


def _residual_emissions(cfg, weights, tokens, expected):
    candidate = EventLM(cfg).eval()
    candidate.load_state_dict(weights)
    candidate = candidate.to_integer()
    emissions, memory_emissions = [], []

    def count_emissions(_module, _inputs, counts):
        emissions.append(counts.abs().sum(dtype=torch.int64).item())

    handles = [module.register_forward_hook(count_emissions)
               for module in candidate.modules() if isinstance(module, _IntegrateFire)]
    try:
        with _IntegerExecution():
            counts = candidate(tokens, memory_pulses=memory_emissions)
            torch.testing.assert_close(counts, expected, rtol=0, atol=0)
    finally:
        for handle in handles:
            handle.remove()
    return (sum(emissions) + sum(value.item() for value in memory_emissions)) / tokens.numel()


@torch.inference_mode()
def _audit(path, device, tokens, frequencies, parent, activity_profile=False, compare_residual_corrections=False):
    model, checkpoint = load_checkpoint(path, torch.device("cpu"))
    parent_seed, parent_weights, parent_counts = parent if parent is not None else (None, None, None)
    if parent_weights is not None and any(
        not torch.equal(tensor, checkpoint["state_dict"][name]) for name, tensor in parent_weights.items()
    ):
        raise AssertionError(f"{path}: local weights differ from the parent")
    integer = model.to_integer()
    emissions = []
    vocabulary_calls = []

    def count_emissions(_module, _inputs, counts):
        emissions.append(counts.abs().sum(dtype=torch.int64).item())

    handles = [module.register_forward_hook(count_emissions)
               for module in integer.modules() if isinstance(module, _IntegrateFire)]
    handles.append(integer.output.register_forward_hook(lambda *_: vocabulary_calls.append(1)))
    activity = {}
    tap_winners = []
    read_changes = []
    read_charge = []
    shared_apertures = {}
    read_transfers = []
    verified_transfers = 0
    if integer.cfg.steering_transfer is not None:
        def check_transfers(module, inputs, output):
            nonlocal verified_transfers
            before = inputs[0].index_select(-1, module.paired_channels).unflatten(-1, (-1, 2))
            after = output.index_select(-1, module.paired_channels).unflatten(-1, (-1, 2))
            if not torch.equal(before.sum(-1, dtype=torch.int32), after.sum(-1, dtype=torch.int32)):
                raise AssertionError("a steering transfer changed the pulse count of a pair")
            verified_transfers += 1

        handles.extend(block.steering_transfer.register_forward_hook(check_transfers) for block in integer.blocks)
    if integer.cfg.trace_read_share_steering:
        def check_read_transfers(module, inputs, output):
            current, original = inputs
            if (output.lt(0) | output.gt(module.request.cap)).any():
                raise AssertionError("shared read aperture exceeds its pulse bounds")
            for before, after in zip(original.split(module.widths, -1), output.split(module.widths, -1), strict=True):
                if not torch.equal(before.sum(-1, dtype=torch.int32), after.sum(-1, dtype=torch.int32)):
                    raise AssertionError("a delay group's spare steering exchange changed its pulse budget")
            threshold = module.request.threshold
            before = current.clamp(-original * threshold, original * threshold)
            after = current.clamp(-output * threshold, output * threshold)
            if (output.lt(original) & before.ne(after)).any():
                raise AssertionError("a steering donor lost read charge")
            moved = (output - original).clamp_min(0).sum(dtype=torch.int64).item()
            recovered = (after - before).abs().sum(dtype=torch.int64).item()
            read_transfers.append((moved, recovered))
            shared_apertures[module] = output

        handles.extend(block.trace.read_drive.steering_share.register_forward_hook(check_read_transfers)
                       for block in integer.blocks)
    if integer.cfg.trace_read_drive is not None:
        def check_driven_read(module, inputs, output):
            base, steering, present, correction = inputs
            aperture = (steering.clamp(max=module.emit.cap) if module.steering_share is None
                        else shared_apertures.pop(module.steering_share))
            if (output.abs() > aperture).any():
                raise AssertionError("a driven trace read exceeded its steering aperture")
            read_changes.append((output - base).abs().sum(dtype=torch.int64).item())
            drive = module.drive_ticks(present, correction)
            limit = aperture * module.emit.threshold
            baseline = base * module.emit.threshold
            passed = drive.clamp(-limit - baseline, limit - baseline)
            magnitude = drive.abs()
            closed = magnitude.masked_fill(aperture.ne(0), 0)
            read_charge.append((magnitude.sum(dtype=torch.int64).item(),
                                passed.abs().sum(dtype=torch.int64).item(), closed.sum(dtype=torch.int64).item()))

        handles.extend(block.trace.read_drive.register_forward_hook(check_driven_read) for block in integer.blocks)
    if integer.cfg.trace_tap_read is not None:
        handles.extend(block.trace_taps.register_forward_hook(
            lambda _module, _inputs, output: tap_winners.append(output[2]),
        ) for block in integer.blocks)
    arrival_activity = []
    if integer.cfg.trace_arrival is not None:
        def arrival_hook(quanta):
            def count_arrival(_module, inputs, output):
                counts = inputs[0]
                if quanta is not None:
                    counts = counts // quanta
                arrival_activity.append((counts.abs().sum(dtype=torch.int64).item(),
                                         output.count_nonzero().item()))
            return count_arrival

        for block in integer.blocks:
            units = (block.packet_thresholds()[:block.arrival_width]
                     if block.packet_thresholds is not None else None)
            handles.append(block.arrival_project.register_forward_hook(arrival_hook(units)))
    if activity_profile:
        populations = {"input": integer.input_emit}
        for index, block in enumerate(integer.blocks):
            populations[f"block_{index}_content"] = block.relay
            populations[f"block_{index}_residual"] = block.emit
            if block.steered:
                populations[f"block_{index}_steering"] = block.steer
            if block.activity_inhibition is not None:
                populations[f"block_{index}_inhibition"] = block.activity_inhibition
            if block.inhibition_redistribution is not None:
                populations[f"block_{index}_inhibition_transfer"] = block.inhibition_redistribution.emit
            if block.steering_transfer is not None:
                populations[f"block_{index}_steering_transfer"] = block.steering_transfer.emit
            if block.late_steering is not None:
                populations[f"block_{index}_late_steering"] = block.late_steering.emit
                populations[f"block_{index}_late_residual"] = block.late_emit
            if block.opponent_steering is not None:
                populations[f"block_{index}_write_protection"] = block.opponent_steering.emit
            if integer.cfg.trace_read_veto:
                populations[f"block_{index}_trace_read_veto"] = block.trace.read_veto.emit
            if integer.cfg.trace_write_confirm:
                populations[f"block_{index}_trace_write_confirm"] = block.trace.write_confirm.emit
            if integer.cfg.trace_integration:
                populations[f"block_{index}_trace_input"] = block.trace.input.emit
            if integer.cfg.trace_read_drive is not None:
                populations[f"block_{index}_trace_read"] = block.trace.read_drive.emit
            if integer.cfg.trace_read_share_steering:
                populations[f"block_{index}_trace_read_requests"] = block.trace.read_drive.steering_share.request
            if integer.cfg.trace_tap_read is not None:
                populations[f"block_{index}_tap_correction"] = block.trace_taps.emit
        for name, module in populations.items():
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, spikes, label=name: activity.__setitem__(label, spikes),
            ))
    memory_stats = {}
    memory_pulses = []
    forecast_spikes = []
    handles.append(integer.forecast_emit.register_forward_hook(
        lambda _module, _inputs, spikes: forecast_spikes.append(spikes.abs().sum(dtype=torch.int64).item()),
    ))
    if integer.fast_synapses is not None:
        integer.fast_synapses.capture = []
    if integer.read_heads is not None:
        integer.read_heads.capture = []
    with _IntegerExecution() as guard:
        counts = integer(tokens, memory_pulses=memory_pulses)
        synapse_activity = head_activity = None
        if integer.fast_synapses is not None:
            synapse_activity, integer.fast_synapses.capture = integer.fast_synapses.capture[0], None
        if integer.read_heads is not None:
            head_activity, integer.read_heads.capture = integer.read_heads.capture[0], None
        for handle in handles:
            handle.remove()
        state = None
        step_counts = []
        for byte in tokens[:, :513].unbind(1):
            predicted, state = integer.step(byte, state)
            step_counts.append(predicted)
        if parent_counts is not None:
            masked, _ = integer.forward_chunk(tokens, hide_age=1)
            torch.testing.assert_close(masked, parent_counts, rtol=0, atol=0)
        if integer.cfg.forecast_key_channels:
            measured, _ = integer.forward_chunk(tokens, stats=memory_stats)
            torch.testing.assert_close(measured, counts, rtol=0, atol=0)
        if integer.cfg.local_trace:
            chunk_state = None
            chunks = []
            for chunk in tokens.split(137, dim=1):
                emitted, chunk_state = integer.forward_chunk(chunk, chunk_state)
                chunks.append(emitted)
            torch.testing.assert_close(torch.cat(chunks, dim=1), counts, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(step_counts, dim=1), counts[:, :513], rtol=0, atol=0)
    accelerated = model.to(device)(tokens.to(device)).cpu()
    torch.testing.assert_close(accelerated, counts, rtol=0, atol=0, check_dtype=False)
    forecast = integer.cfg.route == "steered_correction_code"
    seed = checkpoint["args"]["seed"]
    row = {
        "training_seed": seed,
        "parent_seed": (seed if not integer.has_correction else None) if parent is None else parent_seed,
        "route": integer.cfg.route,
        "checkpoint": str(path),
        "frozen_router_weights_exact": True if parent is not None or not integer.has_correction else None,
        "vocabulary_passes_per_forward": sum(vocabulary_calls),
        "pulses_per_byte": sum(emissions) / tokens.numel(),
        "total_pulses_per_byte": (sum(emissions) + sum(value.item() for value in memory_pulses)) / tokens.numel(),
        "streaming_tensor_bytes": state_bytes(state),
        "native_counts_match_device": str(device),
        "native_batch_matches_step_on_prefix": 513,
        "hidden_reads_match_parent": True if parent is not None else None,
        "integer_operations_checked": guard.operations,
        **(_forecast_errors(integer, tokens, frequencies) if forecast else {}),
    }
    if memory_stats:
        row.update(forecast_key_channels=integer.cfg.forecast_key_channels,
                   match_forecast_key=integer.cfg.match_forecast_key,
                   learn_context_keys=integer.cfg.learn_context_keys,
                   memory_stats={name: int(value.item()) for name, value in memory_stats.items()})
    if integer.cfg.tag_read_competition:
        row["read_competition_gain_ticks"] = integer.prediction_memory.read_inhibition.gains().tolist()
    if integer.cfg.late_steering:
        row["late_steering_gain_ticks"] = [block.late_steering.gains().tolist() for block in integer.blocks]
    if integer.cfg.residual_correction_start is not None:
        row["residual_correction_start"] = integer.cfg.residual_correction_start
        row["ordinary_residual_pulses_per_byte"] = _residual_emissions(
            replace(integer.cfg, residual_correction_start=None), checkpoint["state_dict"], tokens, counts,
        )
    if compare_residual_corrections:
        row["all_residual_corrections_pulses_per_byte"] = _residual_emissions(
            replace(integer.cfg, residual_correction_start=0), checkpoint["state_dict"], tokens, counts,
        )
    if integer.cfg.learn_packet_thresholds:
        row["firing_threshold_ticks"] = [
            (block.packet_thresholds.quanta() * _PACKET_TICKS).tolist() for block in integer.blocks
        ]
    if integer.cfg.delay_window > 1:
        row["delay_window"] = integer.cfg.delay_window
    if integer.cfg.delay_residual:
        row["delay_correction_gain_ticks"] = [block.delay_correction.gains().tolist() for block in integer.blocks]
    if integer.cfg.router_charge_bits != EventLMConfig.router_charge_bits:
        row["router_charge_bits"] = integer.cfg.router_charge_bits
    if integer.cfg.delay_group_widths is not None:
        row["delay_group_widths"] = integer.cfg.group_widths
    if integer.cfg.local_trace:
        row["native_batch_matches_chunk_on_prefix"] = tokens.numel()
    if integer.cfg.opponent_steering:
        row["opponent_active_gains_by_block"] = [
            block.opponent_steering.weight.count_nonzero().item() for block in integer.blocks
        ]
    if integer.cfg.trace_read_veto:
        row["trace_read_veto_gain_ticks"] = [block.trace.read_veto.gains().tolist() for block in integer.blocks]
    if integer.cfg.trace_write_confirm:
        row["trace_write_confirm_gain_ticks"] = [block.trace.write_confirm.gains().tolist() for block in integer.blocks]
    if integer.cfg.trace_integration:
        row["trace_input_nonzero_weights_by_block"] = [
            block.trace.input.weight.count_nonzero().item() for block in integer.blocks
        ]
    if integer.cfg.trace_read_drive is not None:
        if len(read_changes) != len(integer.blocks):
            raise AssertionError("the native audit did not check every driven read population")
        row["trace_read_drive"] = integer.cfg.trace_read_drive
        row["trace_read_drive_bits"] = integer.cfg.trace_read_drive_bits
        row["trace_read_drive_carry"] = integer.cfg.trace_read_drive_carry
        row["trace_read_counts_stayed_within_aperture"] = True
        row["trace_read_drive_active_weights_by_block"] = [
            block.trace.read_drive.weight.count_nonzero().item() for block in integer.blocks
        ]
        if integer.cfg.trace_read_drive.endswith("-rails"):
            row["trace_read_total_variation_cells_by_block"] = [
                block.trace.read_drive.weight.sum(0, dtype=torch.int32).count_nonzero().item()
                for block in integer.blocks
            ]
        row["trace_read_changed_counts_per_byte_by_block"] = [count / tokens.numel() for count in read_changes]
        received, passed, closed = map(sum, zip(*read_charge, strict=True))
        row["trace_read_drive_ticks_per_byte"] = received / tokens.numel()
        row["trace_read_drive_clipped_fraction"] = (received - passed) / received if received else None
        row["trace_read_drive_closed_aperture_fraction"] = closed / received if received else None
    if integer.cfg.trace_read_share_steering:
        if len(read_transfers) != len(integer.blocks) or shared_apertures:
            raise AssertionError("the native audit did not check every shared aperture")
        moved, recovered = map(sum, zip(*read_transfers, strict=True))
        row["trace_read_delay_groups_conserve_steering"] = True
        row["trace_read_steering_donors_preserve_charge"] = True
        row["trace_read_steering_pulses_moved_per_byte"] = moved / tokens.numel()
        row["trace_read_drive_ticks_recovered_per_byte"] = recovered / tokens.numel()
    if integer.cfg.steering_transfer is not None:
        if verified_transfers != len(integer.blocks):
            raise AssertionError("the native audit did not check every steering transfer block")
        row["steering_transfer"] = integer.cfg.steering_transfer
        row["steering_pairs_conserve_pulses"] = True
        row["steering_transfer_active_weights_by_block"] = [
            sum(weight.count_nonzero().item() for weight in block.steering_transfer.weights)
            for block in integer.blocks
        ]
    if forecast_spikes:
        row["forecast_pulses_per_byte"] = sum(forecast_spikes) / tokens.numel()
    if integer.cfg.clock_start is not None:
        row["clock"] = {
            "source": integer.cfg.clock_source, "threshold_forecast_spikes": integer.cfg.clock_threshold,
            "first_clocked_block": integer.cfg.clock_start,
            "ticks_per_byte": chunk_state.clock.events.sum().item() / tokens.numel(),
        }
    if synapse_activity is not None:
        votes, levels, errors = synapse_activity
        memory = integer.fast_synapses
        added = memory.excitation(votes, levels).abs().sum().item()
        # One key neuron fires in every group of every head that has seen its whole context.
        key_spikes = sum(memory.groups * max(tokens.shape[1] - order + 1, 0) for order in memory.orders)
        row["fast_synapses"] = {
            "context_bytes_by_head": list(memory.orders), "groups": memory.groups,
            "key_neurons_per_group": memory.cells, "agreement_levels": memory.levels,
            "store_gate_forecast_spikes": memory.gate,
            "key_neurons": memory.state_cells,
            "readout_spikes_by_head_and_level": memory.readout[:, 1:].tolist(),
            "stored_byte_share": (errors >= memory.gate).float().mean().item(),
            "key_spikes_per_byte": key_spikes / tokens.numel(),
            "recall_spikes_per_byte": added / tokens.numel(),
            "synapse_state_bytes": state.synapses.successor.numel(),
        }
        row["total_pulses_per_byte"] += (key_spikes + added) / tokens.numel()
    if head_activity is not None:
        pred, charge, errors = head_activity
        heads = integer.read_heads
        spikes = heads.activity(pred, charge)
        row["read_heads"] = {
            "context_bytes_by_head": list(heads.orders), "key_neurons_per_head": heads.cells,
            "tape_cells": heads.tape, "store_gate_forecast_spikes": heads.gate,
            "readout_spikes_by_head_and_charge_level": heads.readout[:, 1:].tolist(),
            "stored_byte_share": float((errors >= heads.gate).mean()),
            "live_head_share": spikes["tape"] / (len(heads.orders) * tokens.numel()),
            "key_spikes_per_byte": spikes["key"] / tokens.numel(),
            "tape_spikes_per_byte": spikes["tape"] / tokens.numel(),
            "recall_spikes_per_byte": spikes["recall"] / tokens.numel(),
            "head_state_bytes": state_bytes(state.heads),
        }
        row["total_pulses_per_byte"] += sum(spikes.values()) / tokens.numel()
    if integer.cfg.trace_arrival is not None:
        if len(arrival_activity) != len(integer.blocks):
            raise AssertionError("the audit did not visit every arrival projection")
        row["trace_arrival"] = integer.cfg.trace_arrival
        row["trace_arrival_groups"] = integer.cfg.trace_arrival_groups
        row["trace_arrival_routed_pulses_per_byte_by_block"] = [
            count / tokens.numel() for count, _ in arrival_activity
        ]
        row["trace_arrival_nonzero_currents_per_byte_by_block"] = [
            count / tokens.numel() for _, count in arrival_activity
        ]
    if integer.cfg.learn_inhibition_thresholds:
        row["inhibition_threshold_units_per_base"] = 1 << integer.cfg.inhibition_threshold_bits
        row["inhibition_threshold_quanta_by_block"] = [
            block.inhibition_thresholds.quanta().tolist() for block in integer.blocks
        ]
    if tap_winners:
        row["trace_tap_read"] = integer.cfg.trace_tap_read
        row["trace_tap_release_pulses_per_byte"] = sum(
            winners.ge(0).sum(dtype=torch.int64).item() for winners in tap_winners
        ) / tokens.numel()
        counts_by_block = []
        for block, winners in zip(integer.blocks, tap_winners, strict=True):
            counts_by_group = []
            for delay, count, selected in zip(block.local_delays, block.trace_taps.choices,
                                              winners.unbind(dim=-1), strict=True):
                if count > 1:
                    wins_by_delay = {str(delay - tap): selected.eq(tap).sum(dtype=torch.int64).item()
                                     for tap in range(count)}
                    counts_by_group.append({"anchor_delay": delay, "wins_by_delay": wins_by_delay})
            counts_by_block.append(counts_by_group)
        row["trace_tap_counts_by_block"] = counts_by_block
        row["trace_tap_active_weights_by_block"] = [
            block.trace_taps.weight.count_nonzero().item() for block in integer.blocks
        ]
    if integer.cfg.spread_coincidence:
        row["spread_coincidence_mix_ticks"] = [
            block.spread_coincidence.mix_ticks().tolist() for block in integer.blocks
        ]
    if integer.cfg.delay_filter:
        weights = [kernel.weight for block in integer.blocks for kernel in block.delay_filters if kernel is not None]
        row["delay_filter_parameters"] = sum(weight.numel() for weight in weights)
        row["delay_filter_nonzero_weights"] = sum(int(weight.count_nonzero()) for weight in weights)
    if activity:
        row["activity_profile"] = {}
        if integer.cfg.trace_integration:
            row["trace_input_pulses_per_byte"] = sum(
                activity[f"block_{index}_trace_input"].abs().sum(dtype=torch.int64).item()
                for index in range(len(integer.blocks))
            ) / tokens.numel()
        if integer.cfg.trace_read_share_steering:
            row["trace_read_steering_request_pulses_per_byte"] = sum(
                activity[f"block_{index}_trace_read_requests"].sum(dtype=torch.int64).item()
                for index in range(len(integer.blocks))
            ) / tokens.numel()
        if integer.cfg.steering_transfer is not None:
            row["steering_transfer_pulses_per_byte"] = sum(
                activity[f"block_{index}_steering_transfer"].abs().sum(dtype=torch.int64).item()
                for index in range(len(integer.blocks))
            ) / tokens.numel()
        if tap_winners:
            row["trace_tap_correction_pulses_per_byte"] = sum(
                activity[f"block_{index}_tap_correction"].abs().sum(dtype=torch.int64).item()
                for index in range(len(integer.blocks))
            ) / tokens.numel()
        quantiles = torch.tensor([0.05, 0.5, 0.95])
        for name, packets in activity.items():
            per_byte = packets.abs().float().mean(-1).flatten()
            row["activity_profile"][name] = {
                "mean": per_byte.mean().item(),
                "p05_p50_p95": per_byte.quantile(quantiles).tolist(),
            }
        initialization = checkpoint.get("initialization") or {}
        prior_width = initialization.get("residual_width_before")
        if prior_width is not None:
            row["added_residual_pulses_per_byte"] = {
                name: activity[name][..., prior_width:].abs().sum(dtype=torch.int64).item() / tokens.numel()
                for name in ("input", *(f"block_{index}_residual" for index in range(len(integer.blocks))))
            }
        prior_expansion = initialization.get("expansion_groups_before", initialization.get("expansion_width_before"))
        if prior_expansion is not None:
            added = {}
            for index, block in enumerate(integer.blocks):
                previous = (_delay_group_widths(prior_expansion, len(block.widths))
                            if isinstance(prior_expansion, int) else tuple(prior_expansion))
                for population in ("content", "steering"):
                    name = f"block_{index}_{population}"
                    groups = activity[name].split(block.widths, dim=-1)
                    pulses = sum(group[..., old:].abs().sum(dtype=torch.int64).item()
                                 for group, old in zip(groups, previous, strict=True))
                    added[name] = pulses / tokens.numel()
            row["added_expansion_pulses_per_byte"] = added
        if integer.cfg.route in ("steered", "steered_correction_code"):
            clock = 1 << (2 * integer.cfg.resolution - 1).bit_length()
            fractions = []
            for index in range(len(integer.blocks)):
                content = activity[f"block_{index}_content"].abs()
                steering = activity[f"block_{index}_steering"]
                packed_open = (steering >= content) & (steering < clock)
                total = content.sum(dtype=torch.int64).item()
                opened = content[packed_open].sum(dtype=torch.int64).item()
                fractions.append(opened / total if total else 0.0)
            row["packed_open_fraction_at_partial_clock_by_block"] = fractions
    return row, (seed, checkpoint["state_dict"], counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    models = parser.add_mutually_exclusive_group(required=True)
    models.add_argument("--model-triplet", type=Path, nargs=3, action="append",
                        metavar=("ROUTER", "CONTROL", "CANDIDATE"))
    models.add_argument("--checkpoint", type=Path, nargs="+")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--activity-profile", action="store_true",
                        help="record the distribution of absolute pulse counts per channel by population")
    parser.add_argument("--compare-residual-corrections", action="store_true",
                        help="verify the same counts and measure emissions when every residual uses corrections")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    torch.set_num_threads(4)
    train, validation, test = load_splits(args.enwik8)
    sample = (validation if args.split == "validation" else test)[:4096]
    tokens = torch.from_numpy(sample.astype(np.int64))[None]
    frequencies = torch.from_numpy(np.bincount(train[:1_048_576], minlength=256))
    rows = []
    for parent, control, candidate in args.model_triplet or ():
        row, reference = _audit(parent, device, tokens, frequencies, None, args.activity_profile,
                                args.compare_residual_corrections)
        rows.append(row)
        for path in (control, candidate):
            row, _ = _audit(path, device, tokens, frequencies, reference, args.activity_profile,
                            args.compare_residual_corrections)
            rows.append(row)
    for path in args.checkpoint or ():
        row, _ = _audit(path, device, tokens, frequencies, None, args.activity_profile,
                        args.compare_residual_corrections)
        rows.append(row)
    record = {
        "dataset": "enwik8", "data_sha1": ENWIK8_SHA1, "split": args.split,
        "input_bytes": tokens.numel(), "input_sha256": hashlib.sha256(sample).hexdigest(),
        "input_offset": 0, "method": "Native batch counts at integrate-and-fire outputs and memory write, read and veto circuits.",
        "models": rows,
    }
    with _atomic_output(args.json, "w", overwrite=args.force) as output:
        json.dump(record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
