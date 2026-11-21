"""Compare scorings of the same streams with one reference, overall and by quarter of every stream.

eval_stream.py keeps the bits of every piece of every stream with ``--bits``. For each arm the difference to the
reference is summed over all pieces, and over the pieces of each quarter of the part of a stream that follows its
unscored lead, and divided by the bytes scored there: the gain of the arm, with a 95% interval from resampled
pieces (compare_stream_scores.compare). A gain that grows from the first quarter to the last accumulates with what
the stream has read; one that shrinks fades. The first stream scores its first bytes from an empty state where the
others hear their lead; those bytes count in the overall gain and in no quarter.

Scorings with another lead cut the split into other parts, so only scorings with the same lead are compared:

    python experiments/stream_quarters.py --enwik8 data/enwik8 --split validation --lead 8192 --json q.json \\
        --reference a-8k.json a-8k.npy --arm "taught after training" b-8k.json b-8k.npy \\
        --arm "trained rule" c-8k.json c-8k.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import _atomic_output
from compare_stream_scores import compare
from eval_stream import stream_rows
from reading_curve import piece_bytes
from train_lm import load_splits


def quarters(scored, piece, lead):
    """The quarter, 0 to 3, of its stream's part after the lead that every piece lies in, as [streams, pieces], and
    -1 for a piece of the lead, the first stream's start or the padding. The lead is a whole number of pieces."""
    counts = piece_bytes(scored, piece)
    start = np.arange(counts.shape[1]) * piece
    part = scored.shape[1] - lead
    quarter = np.clip((start - lead) * 4 // part, 0, 3)
    return np.where((start >= lead)[None, :] & (counts > 0), quarter[None, :], -1), counts


def gains(reference, arm, counts, quarter):
    """The arm's gain over the reference in bits per byte, overall and in every quarter, with 95% intervals."""
    targets = int(counts.sum())
    # The gain is the reference's bits less the arm's: compare() returns the second less the first.
    overall, interval = compare(arm, reference, targets)
    rows = []
    for index in range(4):
        inside = quarter == index
        value, spread = compare(arm[inside], reference[inside], int(counts[inside].sum()))
        rows.append({"quarter": index + 1, "gain": value, "gain_95": spread, "bytes": int(counts[inside].sum())})
    return {"gain": overall, "gain_95": interval, "targets": targets, "quarters": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=8192)
    parser.add_argument("--reference", type=Path, nargs=2, required=True, metavar=("RECORD.json", "BITS.npy"))
    parser.add_argument("--arm", nargs=3, action="append", required=True, metavar=("LABEL", "RECORD.json", "BITS.npy"))
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))
    _, _, scored = stream_rows(data, args.streams, args.lead)
    reference = json.loads(args.reference[0].read_text())
    piece = reference["piece_bytes"]
    if args.lead % piece:
        parser.error("the lead must be a whole number of pieces")
    quarter, counts = quarters(scored, piece, args.lead)
    shared = {key: reference.get(key) for key in ("split", "streams", "unscored_lead_bytes", "piece_bytes")}
    if shared != {"split": args.split, "streams": args.streams, "unscored_lead_bytes": args.lead, "piece_bytes": piece}:
        parser.error("the reference does not score these streams")
    first = np.load(args.reference[1])[0]
    record = {**shared, "reference": reference["checkpoints"][0]["checkpoint"],
              "reference_bpb": reference["checkpoints"][0]["bpb"], "arms": []}
    for label, path, bits in args.arm:
        other = json.loads(Path(path).read_text())
        if {key: other.get(key) for key in shared} != shared:
            parser.error(f"{label}: the scoring does not share the reference's streams")
        second = np.load(bits)[0]
        if second.shape != first.shape or first.shape != counts.shape:
            parser.error(f"{label}: the bits do not hold these streams in pieces of {piece} bytes")
        row = {"label": label, "checkpoint": other["checkpoints"][0]["checkpoint"],
               "bpb": other["checkpoints"][0]["bpb"], **gains(first, second, counts, quarter)}
        record["arms"].append(row)
        shown = " ".join(f"{part['gain']:+.4f}" for part in row["quarters"])
        print(f"{label}: {row['bpb']:.4f} bpb | gain {row['gain']:+.4f} "
              f"[{row['gain_95'][0]:+.4f}, {row['gain_95'][1]:+.4f}] | by quarter {shown}", flush=True)
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
