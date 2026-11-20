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
synapses. ``--teach-levels`` gives the output its fast synapses from the
cells after those levels: every arriving byte then teaches them, in each
stream by itself, and no byte is learned from before it is predicted. A
stack trained with fast synapses in its windows (``--plastic-levels`` of the
trainer) runs its trained rule with its trained gains, in whole numbers;
``--plastic-off`` scores it without them. ``--teach-freeze-after`` lets the
fast synapses of every part stop changing after that many bytes, its lead
included, while they keep firing.

With ``--books-dir`` every book of a verified PG-19 manifest is a passage of
its own that starts from an empty state. A book longer than ``--part`` bytes
is cut into equal parts as a split is, each with its lead, so that the
streams are of similar length and run side by side.

A model without such state, as the Transformer, is scored in warmed windows
by eval_warmed_windows.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from eval_stream_lm import load_books
from eval_warmed_windows import PG19_BOOKS
from snnformer import count_params
from snnformer.event_lm import EventLM, EventLMConfig
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


def book_rows(books, part, lead):
    """The inputs and targets of the streams of several books as [streams, bytes], and which targets count.

    Every book is a passage of its own and is scored from its first target. A book of more than ``part`` targets
    is cut into equal parts as ``stream_rows`` cuts a split, and every part after the first hears ``lead``
    unscored bytes of its book. Shorter streams are filled up with zeros that are not scored.
    """
    rows = []
    for book in books:
        parts = -(-(book.numel() - 1) // part)
        rows.append(stream_rows(book, parts, lead if parts > 1 else 0))
    longest = max(inputs.shape[1] for inputs, _, _ in rows)
    return tuple(torch.cat([F.pad(row[kind], (0, longest - row[kind].shape[1])) for row in rows])
                 for kind in range(3))


def lengthen_reach(model, cells, levels=None):
    """Let every read of a stack reach ``cells`` back, or only the reads after ``levels``. The cells beyond a
    read's window take the age offset of the window's oldest cell, as under the setting ``read_reach``."""
    reads = model.state_reads
    if levels is not None and not {str(level) for level in levels} <= set(reads):
        raise ValueError("the stack has no read after one of these levels")
    for level, read in reads.items():
        if levels is not None and int(level) not in levels:
            continue
        if cells < read.window:
            raise ValueError("a read reaches at least its window of cells back")
        read.reach = cells


def forget_by_power(model, slope, levels=None):
    """Give every read of a stack, or the reads after ``levels``, a fixed forgetting slope: ``slope`` units of charge
    for every doubling of the age of a cell beyond the read's window, as a learned slope of ``read_forget`` would
    give. It adds no parameter, and a read whose slope was learned keeps it."""
    reads = model.state_reads
    if levels is not None and not {str(level) for level in levels} <= set(reads):
        raise ValueError("the stack has no read after one of these levels")
    for level, read in reads.items():
        if levels is not None and int(level) not in levels:
            continue
        if read.forget is not None:
            raise ValueError("a read that learned its slope keeps it")
        # The slope counts on top of the address rate, as a learned one does.
        read.forget = torch.full((read.heads,), slope / read.rate, dtype=read.age.dtype, device=read.age.device)
        read.forget_rate = 1.0


@torch.inference_mode()
def score(model, rows, piece, device):
    """The bits of every piece of every stream as [streams, pieces], and the number of scored targets. The rows
    are the inputs, the targets and which of them count, as ``stream_rows`` or ``book_rows`` give them."""
    inputs, targets, scored = rows
    streams = inputs.shape[0]
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
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--enwik8", type=Path, help="canonical enwik8 file, score a complete split")
    dataset.add_argument("--books-dir", type=Path, help="directory of prepared PG-19 excerpts, each book a passage")
    parser.add_argument("--manifest", type=Path, default=PG19_BOOKS, help="manifest of the PG-19 excerpts")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bytes", type=int, help="score only the first bytes of the enwik8 split, for a quick look")
    parser.add_argument("--streams", type=int, default=16, help="parts of the enwik8 split that run side by side")
    parser.add_argument("--part", type=int, default=262144,
                        help="a book of more bytes than this is cut into equal parts that run side by side")
    parser.add_argument("--lead", type=int, default=8192, help="unscored bytes that a part after the first hears")
    parser.add_argument("--piece", type=int, default=512, help="bytes that the stack takes at a time")
    parser.add_argument("--reach", type=int, help="cells that every read reaches back, if more than in training")
    parser.add_argument("--reach-levels", type=int, nargs="+",
                        help="lengthen the reach of the reads after these levels only")
    parser.add_argument("--forget-slope", type=float,
                        help="units of charge that the offset of a cell beyond a read's window takes for every "
                             "doubling of its age, at the reads that take --reach: a fixed power-law forgetting")
    parser.add_argument("--teach-levels", type=int, nargs="+",
                        help="levels whose cells the fast output synapses hear, which every arriving byte teaches")
    parser.add_argument("--teach-threshold", type=int, help="units of fast charge to one spike of an output neuron")
    parser.add_argument("--teach-budget", type=int, default=EventLMConfig.teach_budget,
                        help="teaching spikes of every arriving byte")
    parser.add_argument("--plastic-off", action="store_true",
                        help="score a stack trained with fast output synapses without them, to see how much the "
                             "rest of it leans on them")
    parser.add_argument("--teach-freeze-after", type=int, metavar="BYTES",
                        help="the fast output synapses of every part stop changing after this many bytes, its lead "
                             "included, and keep firing")
    parser.add_argument("--averaged", action="store_true",
                        help="score the averaged weights that a partial checkpoint holds beside its last ones")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--bits", type=Path,
                        help="keep the bits of every piece as [checkpoints, streams, pieces] in this .npy file, so "
                             "that another scoring of the same streams can be compared piece by piece")
    args = parser.parse_args()
    if min(args.streams, args.piece, args.part, args.threads) < 1 or args.lead < 0:
        parser.error("streams, piece, part and threads must be positive and the lead not negative")
    if args.bytes is not None and args.enwik8 is None:
        parser.error("--bytes shortens an enwik8 split")
    if args.reach_levels is not None and args.reach is None:
        parser.error("--reach-levels names the levels that take --reach")
    if args.forget_slope is not None and (args.reach is None or not math.isfinite(args.forget_slope)):
        parser.error("--forget-slope is a finite slope for the cells that --reach adds")
    if (args.teach_levels is None) != (args.teach_threshold is None):
        parser.error("--teach-levels and --teach-threshold go together")
    if args.teach_freeze_after is not None and args.teach_freeze_after < 0:
        parser.error("--teach-freeze-after counts bytes")
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    if args.bits is not None and (args.bits.suffix != ".npy" or args.bits.exists()):
        parser.error("--bits names a new .npy file")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    record = {"protocol": "unbroken_streams", "split": args.split, "streams": args.streams,
              "unscored_lead_bytes": args.lead, "piece_bytes": args.piece, "device": str(device)}
    if args.enwik8 is not None:
        _, validation, test = load_splits(args.enwik8)
        data = torch.from_numpy(np.array(test if args.split == "test" else validation))
        if args.bytes is not None:
            # A shortened split is named in the record, so its score is not taken for the whole split's.
            data = data[:args.bytes]
            record["first_bytes_of_split"] = args.bytes
        streams = stream_rows(data, args.streams, args.lead)
    else:
        manifest = args.manifest.read_bytes()
        books = load_books(manifest, args.books_dir, 0, split=args.split)
        streams = book_rows([book.data for book in books], args.part, args.lead)
        record = {"dataset": "PG-19", "manifest": args.manifest.name,
                  "manifest_sha256": hashlib.sha256(manifest).hexdigest(), "books": len(books),
                  **record, "streams": streams[0].shape[0], "part_bytes": args.part}
    resamples, rows, first, kept = None, [], None, []
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
            lengthen_reach(model, args.reach, args.reach_levels)
        if args.forget_slope is not None:
            try:
                forget_by_power(model, args.forget_slope, args.reach_levels)
            except ValueError as error:
                parser.error(f"{path}: {error}")
        trained = model.plastic_after is not None
        if args.plastic_off:
            if not trained:
                parser.error(f"{path}: --plastic-off silences fast synapses that training ran, and it ran none")
            plastic = {"levels": list(model.plastic_after), "gains": model.plastic_gains.tolist()}
            model.silence_plasticity()
        elif trained:
            if args.teach_levels is not None:
                parser.error(f"{path}: the stack was trained with fast output synapses; give none after training")
            plastic = {"levels": list(model.plastic_after), "rule": model.cfg.plastic_rule,
                       **({"chunk": model.cfg.plastic_chunk} if model.cfg.plastic_rule == "chunked" else {}),
                       "decay_shift": model.cfg.plastic_decay_shift, "threshold": model.cfg.plastic_threshold,
                       "shuffled": model.cfg.plastic_shuffle, "teaching_spikes_per_byte": model.cfg.teach_budget,
                       "gains": model.plastic_gains.tolist()}
        if args.teach_levels is not None:
            model.teach(args.teach_levels, args.teach_threshold, args.teach_budget)
        if args.teach_freeze_after is not None:
            model.freeze_teaching_after(args.teach_freeze_after)
        bits, targets = score(model, streams, args.piece, device)
        reaches = {int(level): read.reach for level, read in model.state_reads.items()}
        row = {"checkpoint": str(path), "run_name": checkpoint.get("run_name"), "parameters": count_params(model),
               # One number where all reads reach equally far, and the reach after every level otherwise.
               "reach_cells": (next(iter(reaches.values())) if len(set(reaches.values())) == 1
                               else dict(sorted(reaches.items()))),
               "training_targets": training_exposure(checkpoint), "targets": targets,
               "bpb": float(bits.sum() / targets)}
        if args.averaged:
            # A partial checkpoint was scored at the average that it holds, not at the weights that its file names.
            row["averaged_weights"] = True
        if args.forget_slope is not None:
            row["forget_slope"] = args.forget_slope
        if args.teach_levels is not None:
            row["taught"] = {"levels": sorted(args.teach_levels), "threshold": args.teach_threshold,
                             "teaching_spikes_per_byte": args.teach_budget}
        if trained:
            row["plastic_silenced" if args.plastic_off else "plastic"] = plastic
        if args.teach_freeze_after is not None:
            row["teach_freeze_after"] = args.teach_freeze_after
        kept.append(bits)
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
    if args.bits is not None:
        np.save(args.bits, np.stack(kept))
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
