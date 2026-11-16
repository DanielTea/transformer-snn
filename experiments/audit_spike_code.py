"""Compare spike emissions and native inference for a router and its prediction banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from benchmark_decode import state_bytes
from common import _atomic_output, resolve_device
from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLMConfig, _IntegrateFire, _PACKET_TICKS
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


@torch.inference_mode()
def _audit(path, device, tokens, frequencies, parent, activity_profile=False):
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
    if activity_profile:
        populations = {"input": integer.input_emit}
        for index, block in enumerate(integer.blocks):
            populations[f"block_{index}_content"] = block.relay
            populations[f"block_{index}_residual"] = block.emit
            if block.steered:
                populations[f"block_{index}_steering"] = block.steer
            if block.activity_inhibition is not None:
                populations[f"block_{index}_inhibition"] = block.activity_inhibition
        for name, module in populations.items():
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, spikes, label=name: activity.__setitem__(label, spikes),
            ))
    memory_stats = {}
    memory_pulses = []
    with _IntegerExecution() as guard:
        counts = integer(tokens, memory_pulses=memory_pulses)
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
    if integer.cfg.local_trace:
        row["native_batch_matches_chunk_on_prefix"] = tokens.numel()
    if activity:
        row["activity_profile"] = {}
        quantiles = torch.tensor([0.05, 0.5, 0.95])
        for name, packets in activity.items():
            per_byte = packets.abs().float().mean(-1).flatten()
            row["activity_profile"][name] = {
                "mean": per_byte.mean().item(),
                "p05_p50_p95": per_byte.quantile(quantiles).tolist(),
            }
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
        row, reference = _audit(parent, device, tokens, frequencies, None, args.activity_profile)
        rows.append(row)
        for path in (control, candidate):
            row, _ = _audit(path, device, tokens, frequencies, reference, args.activity_profile)
            rows.append(row)
    for path in args.checkpoint or ():
        row, _ = _audit(path, device, tokens, frequencies, None, args.activity_profile)
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
