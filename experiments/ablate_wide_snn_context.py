"""Probe the widened SNN on the existing paired enwik8 history windows."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from benchmark_decode import state_bytes
from train_event_lm import load_checkpoint, scoring_logits
from train_lm import ENWIK8_SHA1, load_splits
from snnformer.event_lm import EventLM, _Synapse, _lag, _weights_on_charge_grid


def delay_reader(extra_history, adaptive):
    """Keep the original tap first so it wins ties."""
    def read(block, groups, steering, _pulse_counts):
        delayed = []
        for group, aperture, delay in zip(
            groups, steering.split(block.widths, dim=-1), block.local_delays, strict=True
        ):
            span = delay * (2 if extra_history else 1)
            if not adaptive or span == 1:
                delayed.append(_lag(group, delay))
                continue
            positions = torch.arange(group.shape[1], device=group.device)
            lags = [delay, *range(delay - 1, 0, -1), *range(delay + 1, span + 1)]
            ages = torch.as_tensor(lags, device=group.device)
            packets = torch.stack([_lag(group, lag) for lag in lags], dim=-1)
            query = group.clamp(-aperture, aperture)
            delayed.append(block._match_delay(packets, query, aperture, positions[:, None] >= ages))
        return delayed, None
    return read


def frozen_variant(parent, *, charge_bits=8, grid_bits=None, resolution=16, **changes):
    model = EventLM(replace(parent.cfg, router_charge_bits=charge_bits, resolution=resolution, **changes))
    model.load_state_dict(parent.state_dict())
    if grid_bits is not None:
        for block in model.blocks:
            for synapse in (block.expand, block.project):
                synapse.weight.copy_(_weights_on_charge_grid(synapse.weight, grid_bits))
    return model.eval()


def cache_integer_ticks(model):
    """Use float64 BLAS for exact integer accumulations, including finer grids."""
    model.double()
    model._check_weights = False  # This guard bounds FP32, while the largest here is below 2**31.
    for synapse in model.modules():
        if isinstance(synapse, _Synapse):
            ticks = synapse.weight_ticks()
            synapse.weight_ticks = lambda ticks=ticks: ticks
    return model


def windows(data, starts, history, tail_bytes):
    positions = starts[:, None] + np.arange(-history, tail_bytes)
    return torch.from_numpy(data[positions].astype(np.int64))


def score(model, batches, tail_bytes):
    losses = []
    for batch in batches:
        logits = model(batch[:, :-1])[:, -tail_bytes:]
        if isinstance(model, EventLM):
            logits = scoring_logits(logits, True, model.cfg.resolution)
        targets = batch[:, -tail_bytes:]
        per_byte = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
        losses.append(per_byte.reshape_as(targets).double().sum(1).numpy())
    return np.concatenate(losses) / (tail_bytes * math.log(2))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--test", action="store_true", help="Use the same relative offsets on the test split.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    record = json.loads(args.history_record.read_text())
    snn_path, transformer_path = [Path(row["checkpoint"]) for row in record["models"]]
    starts = np.asarray(record["tail_starts"])
    tail = record["tail_bytes"]
    _, val, test = load_splits(args.data)
    data = test if args.test else val
    device = torch.device("cpu")
    parent, _ = load_checkpoint(snn_path, device)
    assert parent.cfg.router_charge_bits == 8 and parent.cfg.resolution == 16
    transformer, _ = load_checkpoint(transformer_path, device)
    native = parent.to_integer()
    native_bytes = state_bytes(native.step(torch.zeros(1, dtype=torch.long))[1])
    del native

    histories = (1, 32)
    samples = {history: windows(data, starts, history, tail) for history in histories}
    batches = {history: sample.split(48) for history, sample in samples.items()}

    def route(adaptive, extended):
        model = cache_integer_ticks(frozen_variant(parent))
        reader = delay_reader(extended, adaptive)
        for block in model.blocks:
            block._delayed_groups = MethodType(reader, block)
        return model

    def no_bank():
        model = frozen_variant(parent)
        model.recall_synapse.weight.zero_()
        return cache_integer_ticks(model)

    def no_held_charge():
        model = cache_integer_ticks(frozen_variant(parent))

        def read(_trace, args, kwargs, output):
            # Keep the fixed delays, bypass, and trace writes intact.
            steering = args[1]
            delayed = kwargs["delayed_content"]
            return delayed.clamp(-steering, steering), output[1]

        for block in model.blocks:
            block.trace.register_forward_hook(read, with_kwargs=True)
        return model

    axon_bytes = sum(sum(block.delay_sizes) * torch.empty((), dtype=block.axon_dtype).element_size()
                     for block in parent.blocks)

    def phased_bank():
        return cache_integer_ticks(frozen_variant(parent, tag_phase_address=True))

    configurations = [
        ("transformer", lambda: transformer, {}),
        ("snn_baseline", lambda: cache_integer_ticks(frozen_variant(parent)), {"state_bytes": native_bytes}),
        ("routing_adaptive_same_rings", lambda: route(True, False), {"state_bytes": native_bytes}),
        # The batched probe only materializes reads. Unread extra storage is
        # represented by its logical streaming budget in the fixed-read control.
        ("state_double_axons_fixed_read", lambda: route(False, True),
         {"logical_state_bytes": native_bytes + axon_bytes}),
        ("routing_adaptive_double_rings", lambda: route(True, True),
         {"logical_state_bytes": native_bytes + axon_bytes}),
        ("state_bank_56_phase_cells", phased_bank,
         {"state_bytes": state_bytes(frozen_variant(parent, tag_phase_address=True).to_integer()
                                     .step(torch.zeros(1, dtype=torch.long))[1])}),
        ("state_bank_output_disabled", no_bank, {}),
        ("state_held_trace_replaced_by_delayed_coincidence", no_held_charge, {}),
        ("precision_6_router_bits", lambda: cache_integer_ticks(frozen_variant(parent, charge_bits=6)), {}),
        ("precision_10_router_bits_8bit_weights",
         lambda: cache_integer_ticks(frozen_variant(parent, charge_bits=10, grid_bits=8)), {}),
        ("precision_10_router_bits", lambda: cache_integer_ticks(frozen_variant(parent, charge_bits=10)), {}),
        ("precision_8_spike_resolution", lambda: cache_integer_ticks(frozen_variant(parent, resolution=8)), {}),
        ("precision_32_spike_resolution", lambda: cache_integer_ticks(frozen_variant(parent, resolution=32)), {}),
    ]
    reference = cache_integer_ticks(frozen_variant(parent))
    check = samples[32][:16, :-1]
    torch.testing.assert_close(reference(check), parent.to_integer()(check).double(), rtol=0, atol=0)
    for control in (route(False, True), cache_integer_ticks(frozen_variant(parent, charge_bits=10, grid_bits=8))):
        torch.testing.assert_close(control(check), reference(check), rtol=0, atol=0)
    del reference, control

    scores = {}
    results = []
    rng = np.random.default_rng(7907)
    resamples = rng.integers(0, len(starts), size=(4000, len(starts)))
    for label, load, details in configurations:
        model = load()
        scores[label] = {history: score(model, batches[history], tail) for history in histories}
        row = {"condition": label, **details, "histories": []}
        for history in histories:
            values = scores[label][history]
            item = {"warmup_bytes": history, "bpb": float(values.mean())}
            if label != "transformer":
                relative = values - scores["snn_baseline"][history]
                item.update(delta_vs_snn_bpb=float(relative.mean()),
                            paired_bootstrap_95_bpb=np.percentile(relative[resamples].mean(1), (2.5, 97.5)).tolist(),
                            gap_vs_transformer_bpb=float((values - scores["transformer"][history]).mean()))
            row["histories"].append(item)
        print(label, *(f"{entry['bpb']:.6f}" for entry in row["histories"]), flush=True)
        results.append(row)
        del model
    if not args.test:
        for label, expected in zip(("snn_baseline", "transformer"), record["models"], strict=True):
            for row in expected["histories"]:
                history = row["history_bytes"]
                if history in histories:
                    assert abs(scores[label][history].mean() - row["bpb"]) < 1e-6
    for label in ("state_double_axons_fixed_read", "precision_10_router_bits_8bit_weights"):
        for history in histories:
            np.testing.assert_array_equal(scores[label][history], scores["snn_baseline"][history])

    checkpoint_hashes = {}
    for path in (snn_path, transformer_path):
        with path.open("rb") as stream:
            checkpoint_hashes[str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = {
        "split": "test" if args.test else "val", "data_sha1": ENWIK8_SHA1, "device": "cpu",
        "checkpoint_sha256": checkpoint_hashes,
        "history_record": str(args.history_record), "tail_starts": starts.tolist(),
        "tail_bytes": tail, "target_bytes_per_condition": len(starts) * tail,
        "bootstrap": "4000 paired resamples of complete 16-byte tails, seed 7907",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
