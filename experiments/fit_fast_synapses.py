"""Attach surprise-written fast synapses to a trained decoder and fit their readout.

The decoder stays frozen. Its outputs and error spikes are recorded on one
continuous enwik8 split, the synapses are replayed over that stream, and one
strength per head and agreement level is fitted by convex optimisation. The
first half of the split fits and the second half scores every setting. The
chosen setting is then fitted on the whole split and written into a new
checkpoint with integer strengths.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import _atomic_output, resolve_device
from eval_event_lm import _logits_and_targets
from eval_stream_lm import Book
from snnformer.event_lm import EventLM, EventLMConfig
from snnformer.fast_synapses import FastSynapses
from train_event_lm import load_checkpoint, scoring_logits
from train_lm import load_splits

_FORECAST_SPIKES_PER_NAT = 8 / math.log(2)
_CHUNK = 1 << 16


def record_stream(model, checkpoint, data, device, cache, split):
    """Output of every target and the error spikes left by every arriving byte."""
    event = isinstance(model, EventLM)
    targets = len(data) - 1
    stem = None if cache is None else cache / f"{checkpoint['run_name']}_{split}"
    if stem is not None and Path(f"{stem}_outputs.npy").exists() and Path(f"{stem}_surprise.npy").exists():
        return np.load(f"{stem}_outputs.npy", mmap_mode="r"), np.load(f"{stem}_surprise.npy")
    dtype = np.int16 if event else np.float32
    if stem is None:
        outputs = np.zeros((targets, 256), dtype=dtype)
    else:
        cache.mkdir(parents=True, exist_ok=True)
        outputs = np.lib.format.open_memmap(f"{stem}_outputs.npy", mode="w+", dtype=dtype, shape=(targets, 256))
    surprise = np.zeros(targets, dtype=np.int16)
    tensor = torch.from_numpy(data.copy())
    position = 0
    if event:
        # A silent one-neuron memory exposes the forecast neurons' error spikes without changing the output.
        probe = EventLM(replace(model.cfg, synapse_orders=(1,), synapse_groups=1, synapse_cells=2,
                                synapse_levels=1, synapse_gate=0)).eval()
        probe.load_state_dict(model.state_dict(), strict=False)
        probe.to(device)
        probe.fast_synapses.capture = []
        state = None
        with torch.inference_mode():
            for start in range(0, targets, 8192):
                window = tensor[start:start + 8192].unsqueeze(0).to(device=device, dtype=torch.long)
                window = window[:, :targets - start]
                counts, state = probe.forward_chunk(window, state)
                stop = position + window.shape[1]
                outputs[position:stop] = counts[0].cpu().numpy()
                surprise[position:stop] = probe.fast_synapses.capture.pop()[2][0].numpy()
                position = stop
    else:
        options = SimpleNamespace(context=256, history=256, batch_size=8)
        book = Book(0, "enwik8", 0, "", tensor)
        with torch.inference_mode():
            for logits, labels in _logits_and_targets(model, book, options, device, {}):
                logits, labels = logits.flatten(0, 1), labels.flatten()
                stop = position + labels.numel()
                outputs[position:stop] = logits.cpu().numpy()
                lead = logits.amax(-1) - logits.gather(-1, labels[:, None]).squeeze(-1)
                # The same error spikes in the units of the spiking forecast neurons.
                surprise[position + 1:stop + 1] = (lead * _FORECAST_SPIKES_PER_NAT).round().cpu().numpy()[:targets - position - 1]
                position = stop
    if position != targets:
        raise ValueError("recorded targets do not cover the split")
    if stem is not None:
        outputs.flush()
        np.save(f"{stem}_surprise.npy", surprise)
    return outputs, surprise


def replay(memory, data, surprise, outputs, scale):
    """Every byte the fast synapses vote for, with its agreement level per head and the decoder's log probability."""
    targets = len(data) - 1
    heads = len(memory.orders)
    rows, levels, logps, hits = [], [], [], []
    label_logp = np.zeros(targets, dtype=np.float32)
    stored = 0
    state = None
    for start in range(0, targets, _CHUNK):
        stop = min(start + _CHUNK, targets)
        tokens = torch.from_numpy(data[start:stop].astype(np.int64))[None]
        errors = torch.from_numpy(surprise[start:stop].astype(np.int64))[None]
        stored += int((errors >= memory.gate).sum())
        vote, level, state = memory(tokens, errors, state, carry=True)
        # One agreement level per head and byte. A head votes for a byte at most once above level 0.
        dense = torch.zeros((stop - start, heads, 256), dtype=torch.uint8)
        dense.scatter_add_(2, vote[0].clamp_min(0), level[0].to(torch.uint8))
        row, byte = dense.any(1).nonzero(as_tuple=True)
        logp = (torch.from_numpy(np.array(outputs[start:stop])).float() * scale).log_softmax(-1)
        labels = torch.from_numpy(data[start + 1:stop + 1].astype(np.int64))
        rows.append((row + start).numpy().astype(np.int64))
        levels.append(dense[row, :, byte].numpy())
        logps.append(logp[row, byte].numpy())
        hits.append((labels[row] == byte).numpy())
        label_logp[start:stop] = logp.gather(1, labels[:, None])[:, 0].numpy()
    written = float(state.written.float().mean())
    return SimpleNamespace(rows=np.concatenate(rows), levels=np.concatenate(levels), logp=np.concatenate(logps),
                           hits=np.concatenate(hits), label_logp=label_logp, heads=heads,
                           levels_per_head=memory.levels, stored_share=stored / targets, written_share=written)


def loss_nats(features, strengths, start, stop):
    """Summed loss over targets [start, stop) with the given strength per head and level, in nats."""
    low, high = np.searchsorted(features.rows, (start, stop))
    table = torch.cat((strengths.new_zeros(features.heads, 1), strengths), dim=1)
    level = torch.from_numpy(features.levels[low:high]).long()
    boost = table[torch.arange(features.heads), level].sum(1)
    mass = torch.from_numpy(features.logp[low:high]).to(boost.dtype).exp() * torch.expm1(boost)
    rows = torch.from_numpy(features.rows[low:high] - start)
    partition = torch.log1p(boost.new_zeros(stop - start).index_add_(0, rows, mass)).sum()
    matched = boost[torch.from_numpy(features.hits[low:high])].sum()
    return partition - matched - float(features.label_logp[start:stop].sum(dtype=np.float64))


def fit(features, start, stop, iterations=80):
    strengths = torch.zeros(features.heads, features.levels_per_head, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([strengths], lr=1.0, max_iter=iterations, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        value = loss_nats(features, strengths, start, stop) / (stop - start)
        value.backward()
        return value

    optimizer.step(closure)
    return strengths.detach()


def clipped_bpb(memory, data, surprise, outputs, scale, spikes, cap):
    """Loss if the recall spikes had to fit under the vocabulary neurons' own spike limit."""
    probe = FastSynapses(memory.orders, memory.groups, memory.cells, memory.levels, memory.gate)
    probe.readout[:, 1:] = spikes
    targets, nll, state = len(data) - 1, 0.0, None
    for start in range(0, targets, _CHUNK):
        stop = min(start + _CHUNK, targets)
        tokens = torch.from_numpy(data[start:stop].astype(np.int64))[None]
        errors = torch.from_numpy(surprise[start:stop].astype(np.int64))[None]
        votes, levels, state = probe(tokens, errors, state, carry=True)
        counts = torch.from_numpy(np.array(outputs[start:stop])).long() + probe.excitation(votes, levels)[0]
        labels = torch.from_numpy(data[start + 1:stop + 1].astype(np.int64))
        logits = counts.clamp(-cap, cap).float() * scale
        nll += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
    return nll / (targets * math.log(2))


def bpb(features, strengths, start, stop):
    with torch.no_grad():
        return float(loss_nats(features, strengths, start, stop)) / ((stop - start) * math.log(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--orders", default="3,4,6,8,12,16", help="context length of each head in bytes")
    parser.add_argument("--groups", type=int, nargs="+", default=[4])
    parser.add_argument("--cells", type=int, nargs="+", default=[4096], help="key neurons per group")
    parser.add_argument("--levels", type=int, help="agreement levels that fire, all groups when omitted")
    parser.add_argument("--gate", type=int, nargs="+", default=[1],
                        help="error spikes needed to store a byte, 0 stores every byte")
    parser.add_argument("--fit-split", choices=("val",), default="val")
    parser.add_argument("--score-test", action="store_true", help="also score the test split with the final fit")
    parser.add_argument("--cache-dir", type=Path, help="keep the recorded outputs between runs")
    parser.add_argument("--output", type=Path, help="checkpoint with the fitted synapses, needs a single setting")
    parser.add_argument("--training-record", type=Path,
                        help="training record of the source checkpoint, copied beside it for the new checkpoint")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    orders = tuple(int(order) for order in args.orders.split(","))
    settings = list(itertools.product(args.groups, args.cells, args.gate))
    if args.json.expanduser().exists() and not args.force:
        parser.error(f"{args.json} already exists. Choose a new path or use --force to replace it.")
    if (args.output is not None or args.score_test) and len(settings) != 1:
        parser.error("a checkpoint or test score needs a single setting")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    splits = dict(zip(("train", "val", "test"), load_splits(args.enwik8.expanduser()), strict=True))
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser(), device)
    event = isinstance(model, EventLM)
    if args.output is not None and not event:
        parser.error("only a spiking decoder can carry the synapses in its checkpoint")
    if args.training_record is not None and args.output is None:
        parser.error("a training record is copied only for a new checkpoint")
    if event and model.fast_synapses is not None:
        parser.error("the checkpoint already has fast synapses")
    resolution = model.cfg.resolution if event else 1
    scale = float(scoring_logits(torch.ones(()), event, resolution))
    data = np.ascontiguousarray(splits[args.fit_split])
    outputs, surprise = record_stream(model, checkpoint, data, device, args.cache_dir, args.fit_split)
    targets = len(data) - 1
    half = targets // 2

    def strengths_as_stored(strengths):
        """Spiking decoders keep whole vocabulary spikes."""
        return (strengths / scale).round() * scale if event else strengths

    records = []
    for groups, cells, gate in settings:
        started = time.time()
        memory = FastSynapses(orders, groups, cells, groups if args.levels is None else args.levels, gate)
        features = replay(memory, data, surprise, outputs, scale)
        zero = torch.zeros(features.heads, features.levels_per_head, dtype=torch.float64)
        held_out = strengths_as_stored(fit(features, 0, half))
        record = {
            "orders": list(orders), "groups": groups, "cells_per_group": cells, "levels": memory.levels, "gate": gate,
            "key_neurons": memory.state_cells, "synapse_state_bytes": memory.state_cells + memory.longest,
            "stored_byte_share": features.stored_share, "key_neurons_written_share": features.written_share,
            "held_out": {"fit_targets": half, "scored_targets": targets - half,
                         "base_bpb": bpb(features, zero, half, targets),
                         "bpb": bpb(features, held_out, half, targets)},
        }
        record["held_out"]["gain_bpb"] = record["held_out"]["base_bpb"] - record["held_out"]["bpb"]
        print(f"groups={groups} cells={cells} gate={gate} state={record['synapse_state_bytes']:,} B "
              f"stored={features.stored_share:.3f}: held-out {record['held_out']['base_bpb']:.4f} -> "
              f"{record['held_out']['bpb']:.4f} (gain {record['held_out']['gain_bpb']:.4f}, "
              f"{time.time() - started:.0f}s)", flush=True)
        if args.output is not None or args.score_test:
            final = strengths_as_stored(fit(features, 0, targets))
            record["fit"] = {"split": args.fit_split, "targets": targets, "base_bpb": bpb(features, zero, 0, targets),
                             "bpb": bpb(features, final, 0, targets),
                             "strength_nats": final.tolist(),
                             **({"strength_spikes": (final / scale).round().to(torch.int64).tolist()} if event else {})}
            if event:
                spikes = (final / scale).round().to(torch.int32)
                cap = model.output_emit.cap
                if spikes.clamp_min(0).amax(1).sum() > cap:
                    parser.error("the fitted strengths exceed the spike limit of a recall neuron")
                # The recall neurons are counted beside the vocabulary neurons. This is the loss with one shared limit.
                record["fit"]["bpb_with_recall_clipped_at_vocabulary_limit"] = clipped_bpb(
                    memory, data, surprise, outputs, scale, spikes, cap)
                print(f"fit {record['fit']['base_bpb']:.5f} -> {record['fit']['bpb']:.5f} "
                      f"(clipped at the vocabulary limit {record['fit']['bpb_with_recall_clipped_at_vocabulary_limit']:.5f})",
                      flush=True)
            if args.score_test:
                test = np.ascontiguousarray(splits["test"])
                test_outputs, test_surprise = record_stream(model, checkpoint, test, device, args.cache_dir, "test")
                scored = replay(memory, test, test_surprise, test_outputs, scale)
                record["test"] = {"targets": len(test) - 1, "base_bpb": bpb(scored, zero, 0, len(test) - 1),
                                  "bpb": bpb(scored, final, 0, len(test) - 1),
                                  "stored_byte_share": scored.stored_share}
                print(f"test {record['test']['base_bpb']:.5f} -> {record['test']['bpb']:.5f}", flush=True)
            if args.output is not None:
                cfg = replace(model.cfg, synapse_orders=orders, synapse_groups=groups, synapse_cells=cells,
                              synapse_levels=memory.levels, synapse_gate=gate)
                attached = EventLM(cfg).eval()
                attached.load_state_dict(checkpoint["state_dict"], strict=False)
                attached.fast_synapses.readout[:, 1:] = (final / scale).round().to(torch.int32)
                saved = dict(checkpoint)
                saved["cfg"] = {**checkpoint["cfg"], "synapse_orders": orders, "synapse_groups": groups,
                                "synapse_cells": cells, "synapse_levels": memory.levels, "synapse_gate": gate}
                saved["state_dict"] = attached.state_dict()
                saved["run_name"] = f"{checkpoint['run_name']}_fast_synapses"
                with args.checkpoint.expanduser().open("rb") as stream:
                    source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
                saved["fast_synapses"] = {"source_checkpoint": str(args.checkpoint), "source_sha256": source_hash,
                                          "fit_split": f"enwik8 {args.fit_split}", "fit_targets": targets}
                args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
                torch.save(saved, args.output.expanduser())
                record["checkpoint"] = str(args.output)
                if args.training_record is not None:
                    # The gradient training is the source run's. Only the checkpoint and its synapses differ.
                    training = json.loads(args.training_record.expanduser().read_text(encoding="utf-8"))
                    if training["args"]["checkpoint"] != str(args.checkpoint):
                        parser.error(f"{args.training_record} belongs to a different checkpoint")
                    training["args"] = {**training["args"], "checkpoint": str(args.output)}
                    training["cfg"] = {**training["cfg"], **{name: value for name, value in saved["cfg"].items()
                                                             if name.startswith("synapse_")}}
                    training["cfg"]["synapse_orders"] = list(orders)
                    # The fit record is kept beside the training record.
                    training["fast_synapses"] = {**saved["fast_synapses"], "fit_record": args.json.name}
                    derived = args.training_record.expanduser().with_name(f"{saved['run_name']}.json")
                    with _atomic_output(derived, "w", args.force) as stream:
                        json.dump(training, stream, indent=2)
                        stream.write("\n")
        records.append(record)
    report = {"source_checkpoint": str(args.checkpoint), "run_name": checkpoint["run_name"],
              "model_type": checkpoint["model_type"], "fit_split": args.fit_split, "settings": records}
    with _atomic_output(args.json.expanduser(), "w", args.force) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
