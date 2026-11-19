"""Score a stack of reads and cells on a complete enwik8 split as a stream.

The split is cut into ``--streams`` equal parts that run side by side. Each
part is one unbroken stream: the stack takes it in pieces of ``--piece``
bytes and carries the state of its reads and cells from piece to piece. No
window starts from an empty state, and every target of the split is scored
once, with all of its part behind it. A part after the first hears the
``--lead`` bytes before it first, unscored, so only the first bytes of the
split are predicted from an empty state.

The spikes do not depend on ``--piece``: a stream cut anywhere fires the
spikes of one pass over the whole of it. ``--reach`` lets every read reach
further back than it did in training, which a stream allows at no cost in
synapses.

A model without such state, as the Transformer, is scored in warmed windows
by eval_warmed_windows.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from snnformer import count_params
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits, training_exposure
from train_lm import load_splits


def stream_rows(data, streams, lead):
    """The inputs and targets of every stream as [streams, bytes], and which targets count.

    The first stream starts with the split and is scored from its first target. Every other stream starts
    ``lead`` targets before its part, and those are not scored. The last stream is filled up with zeros that are
    not scored either.
    """
    targets = data.numel() - 1
    if targets <= lead + streams:
        raise ValueError("the split is too short for these streams and their lead")
    part = -(-(targets - lead) // streams)
    places = torch.arange(streams)[:, None] * part + torch.arange(lead + part)
    scored = places < targets
    scored[1:, :lead] = False
    places = places.clamp_max(targets - 1)
    return data[places].long(), data[places + 1].long(), scored


def lengthen_reach(model, cells):
    """Let every read of a stack reach ``cells`` back. The cells beyond its window take the age offset of the
    window's oldest cell, as under the setting ``read_reach``."""
    for read in model.state_reads.values():
        if cells < read.window:
            raise ValueError("a read reaches at least its window of cells back")
        read.reach = cells


@torch.inference_mode()
def score(model, data, streams, lead, piece, device):
    """The bits of every piece of every stream as [streams, pieces], and the number of scored targets."""
    inputs, targets, scored = stream_rows(data, streams, lead)
    resolution = model.cfg.resolution
    bits, state = [], None
    for start in range(0, inputs.shape[1], piece):
        span = slice(start, start + piece)
        counts, state = model.stream(inputs[:, span].to(device), state)
        if not scored[:, span].any():
            continue
        logits = scoring_logits(counts, True, resolution)
        losses = F.cross_entropy(logits.flatten(0, 1), targets[:, span].to(device).flatten(), reduction="none")
        losses = losses.view(streams, -1).cpu() * scored[:, span]
        # MPS has no 64-bit floats, so the sum is taken on the CPU.
        bits.append(losses.sum(1, dtype=torch.float64) / math.log(2))
    return torch.stack(bits, 1).numpy(), int(scored.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True, help="canonical enwik8 file")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bytes", type=int, help="score only the first bytes of the split, for a quick look")
    parser.add_argument("--streams", type=int, default=16, help="parts of the split that run side by side")
    parser.add_argument("--lead", type=int, default=8192, help="unscored bytes that a part after the first hears")
    parser.add_argument("--piece", type=int, default=512, help="bytes that the stack takes at a time")
    parser.add_argument("--reach", type=int, help="cells that every read reaches back, if more than in training")
    parser.add_argument("--averaged", action="store_true",
                        help="score the averaged weights that a partial checkpoint holds beside its last ones")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.streams, args.piece, args.threads) < 1 or args.lead < 0:
        parser.error("streams, piece and threads must be positive and the lead not negative")
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))
    record = {"protocol": "unbroken_streams", "split": args.split, "streams": args.streams,
              "unscored_lead_bytes": args.lead, "piece_bytes": args.piece, "device": str(device)}
    if args.bytes is not None:
        # A shortened split is named in the record, so its score is not taken for the whole split's.
        data = data[:args.bytes]
        record["first_bytes_of_split"] = args.bytes
    resamples, rows, first = None, [], None
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if not isinstance(model, EventLM) or not model.cfg.stack:
            parser.error(f"{path}: a stream is scored on a stack of reads and cells")
        if args.averaged:
            if "averaged_parameters" not in checkpoint:
                parser.error(f"{path}: the checkpoint holds no averaged weights")
            model.load_state_dict(checkpoint["averaged_parameters"], strict=False)
            # The average of weights that were each in the range of exact sums can leave it.
            model.synaptic_scaling()
        if args.reach is not None:
            lengthen_reach(model, args.reach)
        bits, targets = score(model, data, args.streams, args.lead, args.piece, device)
        row = {"checkpoint": str(path), "run_name": checkpoint.get("run_name"), "parameters": count_params(model),
               "reach_cells": next(iter(model.state_reads.values())).reach,
               "training_targets": training_exposure(checkpoint), "targets": targets,
               "bpb": float(bits.sum() / targets), "averaged_weights": args.averaged}
        # The pieces are resampled together for the difference to the first checkpoint.
        paired = bits.reshape(-1)
        if first is None:
            first = paired
            resamples = np.random.default_rng(5261).integers(len(paired), size=(3000, len(paired)))
        else:
            difference = (paired - first)[resamples].sum(axis=1) / targets
            row["bpb_over_first"] = float((paired - first).sum() / targets)
            row["bpb_over_first_paired_95"] = np.percentile(difference, (2.5, 97.5)).tolist()
        rows.append(row)
        extra = ("" if "bpb_over_first" not in row else
                 f" | {row['bpb_over_first']:+.4f} to the first "
                 f"[{row['bpb_over_first_paired_95'][0]:+.4f}, {row['bpb_over_first_paired_95'][1]:+.4f}]")
        print(f"{path} | {row['parameters']:,} parameters | {targets:,} targets | {row['bpb']:.6f} bpb{extra}",
              flush=True)
        del model
    record["checkpoints"] = rows
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
