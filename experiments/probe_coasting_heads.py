"""Fit-only test of coasting read heads on a frozen decoder.

The decoder's recorded outputs and error spikes come from the cache that
``fit_fast_synapses.py`` writes. Read heads are replayed over the same
continuous enwik8 stream, one strength per head and charge level is fitted by
the same convex fit as the fast synapses, and the held-out half of validation
scores every setting. Nothing in the decoder changes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from common import resolve_device
from fit_fast_synapses import bpb, fit, record_stream
from snnformer.coasting_heads import LEVELS, SEED, charge_level, coast
from snnformer.event_lm import EventLM
from snnformer.fast_synapses import FastSynapses
from train_event_lm import load_checkpoint, scoring_logits
from train_lm import load_splits

_CHUNK = 1 << 16
DEFAULTS = dict(orders="3,4,6,8,12,16", cells=4096, gate=1, tape=8192, lookup=0, miss=0, relock=0, seed=SEED,
                lifetime=0)


def load_stream(cache, run_name, split, data, scale, spiking=True):
    stem = cache / f"{run_name}_{split}"
    outputs = np.load(f"{stem}_outputs.npy", mmap_mode="r")
    surprise = np.ascontiguousarray(np.load(f"{stem}_surprise.npy"))
    targets = len(data) - 1
    partition, label = Path(f"{stem}_partition.npy"), Path(f"{stem}_label_logp.npy")
    if not partition.exists() or not label.exists():
        log_partition = np.zeros(targets, dtype=np.float32)
        label_logp = np.zeros(targets, dtype=np.float32)
        for start in range(0, targets, _CHUNK):
            stop = min(start + _CHUNK, targets)
            logits = torch.from_numpy(np.array(outputs[start:stop])).float() * scale
            total = logits.logsumexp(-1)
            labels = torch.from_numpy(data[start + 1:stop + 1].astype(np.int64))
            log_partition[start:stop] = total.numpy()
            label_logp[start:stop] = (logits.gather(1, labels[:, None])[:, 0] - total).numpy()
        np.save(partition, log_partition)
        np.save(label, label_logp)
    return SimpleNamespace(data=np.ascontiguousarray(data), outputs=outputs, surprise=surprise, scale=scale, spiking=spiking,
                           targets=targets, partition=np.load(partition), label_logp=np.load(label))


def synapse_votes(stream, orders=(3, 4, 6, 8, 12, 16), groups=4, cells=1024, gate=1):
    """Every byte the v11 fast synapses vote for with the agreement level of each head."""
    memory = FastSynapses(orders, groups, cells, groups, gate)
    rows, bytes_, levels, state = [], [], [], None
    for start in range(0, stream.targets, _CHUNK):
        stop = min(start + _CHUNK, stream.targets)
        tokens = torch.from_numpy(stream.data[start:stop].astype(np.int64))[None]
        errors = torch.from_numpy(stream.surprise[start:stop].astype(np.int64))[None]
        vote, level, state = memory(tokens, errors, state, carry=True)
        dense = torch.zeros((stop - start, len(orders), 256), dtype=torch.uint8)
        dense.scatter_add_(2, vote[0].clamp_min(0), level[0].to(torch.uint8))
        row, byte = dense.any(1).nonzero(as_tuple=True)
        rows.append((row + start).numpy())
        bytes_.append(byte.numpy())
        levels.append(dense[row, :, byte].numpy())
    return [(np.concatenate(rows), np.concatenate(bytes_), np.concatenate(levels))]


def head_votes(stream, setting):
    """The byte each coasting head predicts with its charge level, one part per head."""
    parts, stats = [], []
    for order in (int(order) for order in str(setting["orders"]).split(",")):
        pred, charge, counts = coast(stream.data, stream.surprise, order, int(setting["cells"]), int(setting["gate"]),
                                     int(setting["tape"]), int(setting["lookup"]), int(setting["miss"]),
                                     int(setting["relock"]), int(setting["seed"]), int(setting["lifetime"]))
        rows = np.nonzero(pred >= 0)[0]
        level = charge_level(charge[rows]).astype(np.uint8)
        parts.append((rows, pred[rows].astype(np.int64), level[:, None]))
        hit = stream.data[rows + 1] == pred[rows]
        stats.append({"order": order, "synapse_moves": int(counts[0]), "lookups": int(counts[1]),
                      "placements": int(counts[2]), "live_share": float(counts[3] / stream.targets),
                      "hit_share_when_live": float(hit.mean()) if len(rows) else 0.0,
                      "hit_share_by_level": [float(hit[level == value].mean()) if (level == value).any() else None
                                             for value in range(1, LEVELS + 1)],
                      "rows_by_level": [int((level == value).sum()) for value in range(1, LEVELS + 1)]})
    return parts, stats


def features(stream, parts):
    """Join the votes of all heads into one row per target and voted byte."""
    heads = sum(part[2].shape[1] for part in parts)
    keys = np.concatenate([part[0].astype(np.int64) * 256 + part[1] for part in parts])
    unique, inverse = np.unique(keys, return_inverse=True)
    levels = np.zeros((len(unique), heads), dtype=np.uint8)
    offset = position = 0
    for rows, _, level in parts:
        index = inverse[position:position + len(rows)]
        levels[index, offset:offset + level.shape[1]] = level
        offset += level.shape[1]
        position += len(rows)
    rows, byte = unique >> 8, unique & 255
    logp = np.empty(len(rows), dtype=np.float32)
    for start in range(0, len(rows), 1 << 20):
        chunk = slice(start, start + (1 << 20))
        logp[chunk] = stream.outputs[rows[chunk], byte[chunk]].astype(np.float32) * stream.scale - stream.partition[rows[chunk]]
    return SimpleNamespace(rows=rows, levels=levels, logp=logp, hits=stream.data[rows + 1] == byte,
                           label_logp=stream.label_logp, heads=heads, levels_per_head=LEVELS)


def stored(strengths, stream):
    """Whole recall spikes, as the decoder would keep them. Full attention has no spikes and keeps real strengths."""
    return (strengths / stream.scale).round() * stream.scale if stream.spiking else strengths


def score(stream, parts, *, test=None, test_parts=None, full_fit=False):
    half = stream.targets // 2
    table = features(stream, parts)
    zero = torch.zeros(table.heads, LEVELS, dtype=torch.float64)
    held_out = stored(fit(table, 0, half), stream)
    record = {"heads": table.heads, "voted_bytes_per_target": len(table.rows) / stream.targets,
              "held_out_base_bpb": bpb(table, zero, half, stream.targets),
              "held_out_bpb": bpb(table, held_out, half, stream.targets)}
    record["held_out_gain_bpb"] = record["held_out_base_bpb"] - record["held_out_bpb"]
    record["held_out_strength_spikes"] = (held_out / stream.scale).round().to(torch.int64).tolist()
    if test is not None or full_fit:
        final = stored(fit(table, 0, stream.targets), stream)
        record["fit_base_bpb"] = bpb(table, zero, 0, stream.targets)
        record["fit_bpb"] = bpb(table, final, 0, stream.targets)
        record["strength_spikes"] = (final / stream.scale).round().to(torch.int64).tolist()
    if test is not None:
        scored = features(test, test_parts)
        record["test_base_bpb"] = bpb(scored, zero, 0, test.targets)
        record["test_bpb"] = bpb(scored, final, 0, test.targets)
        record["test_gain_bpb"] = record["test_base_bpb"] - record["test_bpb"]
    return record


def write_readout(folder, index, record, setting, run_name, synapse_heads):
    """The fitted strengths in the form the trainer reads with --head-readout."""
    strengths = record["strength_spikes"]
    fitted = {"source_run": run_name, "fit": "enwik8 validation, all targets",
              "setting": {**{name: int(setting[name]) for name in ("cells", "gate", "tape", "lookup", "miss", "relock")},
                          "orders": [int(order) for order in str(setting["orders"]).split(",")]},
              "head_strength_spikes": strengths[synapse_heads:]}
    if synapse_heads:
        # The first heads of a joint fit are the fast synapses with their four agreement levels.
        fitted["synapse_strength_spikes"] = [row[:4] for row in strengths[:synapse_heads]]
    name = "joint" if synapse_heads else "heads"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}_{index}.json").write_text(json.dumps(fitted, indent=2) + "\n")


def parse(text):
    setting = dict(DEFAULTS)
    for item in filter(None, text.split(";")):
        name, value = item.split("=")
        if name not in setting:
            raise ValueError(f"unknown head setting {name}")
        setting[name] = value if name == "orders" else int(value)
    return setting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="the frozen decoder, read for its spike scale")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True, help="keeps the compiled replay")
    parser.add_argument("--setting", action="append", default=[],
                        help="head settings as name=value;name=value, for example 'tape=2048;miss=2'")
    parser.add_argument("--synapses", choices=("alone", "joint", "none"), nargs="+", default=["none"],
                        help="score the v11 fast synapses alone, together with the heads, or not at all")
    parser.add_argument("--synapse-orders", default="3,4,6,8,12,16", help="context lengths of the v11 fast synapses")
    parser.add_argument("--score-test", action="store_true")
    parser.add_argument("--readout-dir", type=Path,
                        help="write the strengths fitted on all of validation as heads_<n>.json and joint_<n>.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu",
                        help="records the decoder when the cache has no outputs for it")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("SNN_BUILD_DIR", str(args.work_dir.expanduser()))
    splits = dict(zip(("train", "val", "test"), load_splits(args.enwik8.expanduser()), strict=True))
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser(), "cpu")
    event = isinstance(model, EventLM)
    scale = float(scoring_logits(torch.ones(()), event, model.cfg.resolution if event else 1))
    names = ("val", "test") if args.score_test else ("val",)
    missing = [name for name in names
               if not (args.cache_dir / f"{checkpoint['run_name']}_{name}_surprise.npy").exists()]
    if missing:
        # The recorded outputs are the decoder's own. Its synapses and heads are replayed afterwards.
        plain = model
        if event:
            plain = EventLM(replace(model.cfg, synapse_orders=None, head_orders=None)).eval()
            plain.load_state_dict({name: value for name, value in model.state_dict().items()
                                   if not name.startswith(("fast_synapses.", "read_heads."))})
        plain.to(device)
        for name in missing:
            record_stream(plain, checkpoint, np.ascontiguousarray(splits[name]), device, args.cache_dir, name)
        del plain
    del model
    val = load_stream(args.cache_dir, checkpoint["run_name"], "val", splits["val"], scale, event)
    test = (load_stream(args.cache_dir, checkpoint["run_name"], "test", splits["test"], scale, event)
            if args.score_test else None)
    report = {"run_name": checkpoint["run_name"], "spiking": event, "spikes_per_nat": 1 / scale, "records": []}
    synapses = test_synapses = None
    synapse_orders = tuple(int(order) for order in args.synapse_orders.split(","))
    if {"alone", "joint"} & set(args.synapses):
        started = time.time()
        synapses = synapse_votes(val, synapse_orders)
        test_synapses = synapse_votes(test, synapse_orders) if test is not None else None
        if "alone" in args.synapses:
            record = {"name": "v11 fast synapses", "setting": f"orders={args.synapse_orders}", **score(val, synapses, test=test, test_parts=test_synapses,
                                                           full_fit=args.readout_dir is not None)}
            report["records"].append(record)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items() if "strength" not in k}), f"{time.time() - started:.0f}s",
                  flush=True)
    for index, text in enumerate(args.setting):
        started = time.time()
        setting = parse(text)
        parts, stats = head_votes(val, setting)
        test_parts = head_votes(test, setting)[0] if test is not None else None
        variants = [("heads", parts, test_parts)]
        if "joint" in args.synapses:
            variants.append(("synapses and heads", synapses + parts,
                             None if test is None else test_synapses + test_parts))
        for name, votes, test_votes in variants:
            record = {"name": name, "setting": text, "parsed": setting,
                      **score(val, votes, test=test, test_parts=test_votes, full_fit=args.readout_dir is not None)}
            if args.readout_dir is not None:
                write_readout(args.readout_dir.expanduser(), index, record, setting, checkpoint["run_name"],
                              0 if name == "heads" else len(synapse_orders))
            if name == "heads":
                record["head_stats"] = stats
            report["records"].append(record)
            args.json.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({k: v for k, v in record.items()
                              if "strength" not in k and k not in ("head_stats", "parsed")}),
                  f"{time.time() - started:.0f}s", flush=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
