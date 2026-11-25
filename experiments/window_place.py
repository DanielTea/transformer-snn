"""The bits of a stack by a byte's place in a window of its stream, and the paired difference of two stacks.

A sleeping stack (--stack-sleep WINDOW) replays every window of WINDOW bytes of a stream, counted from the stream's
first byte, and the bytes after a window read the cells that its replay wrote. A byte early in a window reads
mostly replayed cells, a byte late in it mostly the cells of its own window, which no replay has touched yet. This
scores stacks on an enwik8 split as eval_stream.py does (the split cut into unbroken streams that run side by side,
each after its lead) and sums every scored byte's bits by its place in the window, the place counted from its
stream's first byte. Two scorings of the same streams pair place by place: the difference of the second minus the
first at every place, and at the first and the last quarter of the window.

usage: python experiments/window_place.py --checkpoint BASE.pt ARM.pt --enwik8 data/enwik8 --split validation \\
    --bytes 2000000 --window 64 --json place.json [--device cuda]
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
from eval_stream import stream_rows
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits
from train_lm import load_splits


@torch.inference_mode()
def bits_by_place(model, rows, piece, device, window):
    """The bits of every place in the window summed over all scored bytes, and the number of scored bytes there."""
    inputs, targets, scored = rows
    streams = inputs.shape[0]
    sums = torch.zeros(window, dtype=torch.float64)
    counts = torch.zeros(window, dtype=torch.float64)
    state = None
    for start in range(0, inputs.shape[1], piece):
        span = slice(start, start + piece)
        output, state = model.stream(inputs[:, span].to(device), state)
        if not scored[:, span].any():
            continue
        logits = scoring_logits(output, True, model.cfg.resolution)
        losses = F.cross_entropy(logits.flatten(0, 1), targets[:, span].to(device).flatten(), reduction="none")
        losses = losses.view(streams, -1).cpu().double() * scored[:, span]
        place = torch.arange(start, start + losses.shape[1]) % window
        sums.index_add_(0, place, losses.sum(0) / math.log(2))
        counts.index_add_(0, place, scored[:, span].sum(0).double())
    return sums, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, help="score only the first bytes of the split")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=8192)
    parser.add_argument("--piece", type=int, default=512)
    parser.add_argument("--window", type=int, default=64, help="the sleep window, counted from a stream's first byte")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))
    if args.bytes is not None:
        data = data[:args.bytes]
    rows = stream_rows(data, args.streams, args.lead)
    record = {"protocol": "unbroken_streams", "split": args.split, "streams": args.streams,
              "unscored_lead_bytes": args.lead, "piece_bytes": args.piece, "window": args.window,
              **({"first_bytes_of_split": args.bytes} if args.bytes is not None else {}), "scorings": []}
    places = []
    for path in args.checkpoint:
        model, _ = load_checkpoint(path, device)
        if not isinstance(model, EventLM) or not model.cfg.stack:
            parser.error(f"{path}: a stream is scored on a stack of reads and cells")
        model.eval()
        sums, counts = bits_by_place(model, rows, args.piece, device, args.window)
        places.append((sums, counts))
        record["scorings"].append({"checkpoint": path.name, "run": path.parent.name, "sleep": model.cfg.stack_sleep,
                                   "bpb": float(sums.sum() / counts.sum()),
                                   "bpb_by_place": (sums / counts).tolist()})
        print(f"{path.parent.name}: {float(sums.sum() / counts.sum()):.6f} bpb over {int(counts.sum()):,} targets",
              flush=True)
    if len(places) == 2:
        (first, counts), (second, _) = places
        difference = (second - first) / counts
        quarter = args.window // 4
        record["second_minus_first_by_place"] = difference.tolist()
        record["second_minus_first_first_quarter"] = float((second - first)[:quarter].sum() / counts[:quarter].sum())
        record["second_minus_first_last_quarter"] = float((second - first)[-quarter:].sum() / counts[-quarter:].sum())
    with _atomic_output(args.json, "w", False) as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
