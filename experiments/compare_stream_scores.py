"""Compare two scorings of the same streams piece by piece.

eval_stream.py and eval_taught_reference.py keep the bits of every piece of
every stream with ``--bits``. Two such scorings of one split, cut into the
same streams with the same lead, score the same targets. Their difference is
then resampled over pieces, which removes the spread that both share: how
hard a piece of text is. So a stack can be compared with a Transformer, or
with itself under another reach, on far fewer bytes than two separate
intervals would need.

Pieces of two lengths are compared at the longer one, which must be a
multiple of the shorter. The interval covers the choice of the scored
pieces. It does not cover the training run: each scoring is one seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import _atomic_output

# What two scorings must share to score the same targets of the same streams.
SHARED = ("dataset", "manifest_sha256", "split", "streams", "unscored_lead_bytes", "part_bytes", "first_bytes_of_split")


def scoring(record, bits, row):
    """The bits of one scoring as [streams, pieces], the bytes of a piece, the scored targets and its name."""
    if "checkpoints" in record:
        # eval_stream.py: one row for every checkpoint.
        entry = record["checkpoints"][row]
        return bits[row], record["piece_bytes"], entry["targets"], entry["checkpoint"]
    # eval_taught_reference.py: one row for every rate of the fast output synapses.
    entry = record["fast_output_synapses"][row]
    return bits[row], record["context_bytes"], record["targets"], f"{record['checkpoint']} at rate {entry['rate']:g}"


def regroup(bits, by):
    """Every ``by`` neighbouring pieces of each stream summed into one. The last one may hold fewer."""
    return np.add.reduceat(bits, np.arange(0, bits.shape[-1], by), axis=-1)


def compare(first, second, targets, resamples=3000, seed=5261):
    """Bits per byte of the second scoring over the first, and the 95% interval of that under resampled pieces."""
    paired = (second - first).reshape(-1)
    draws = np.random.default_rng(seed).integers(len(paired), size=(resamples, len(paired)))
    return float(paired.sum() / targets), np.percentile(paired[draws].sum(axis=1) / targets, (2.5, 97.5)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("first", "second"):
        parser.add_argument(f"--{name}", type=Path, nargs=2, required=True, metavar=("RECORD.json", "BITS.npy"),
                            help=f"the {name} scoring: its record and the bits that --bits kept")
        parser.add_argument(f"--{name}-row", type=int, default=0,
                            help="which checkpoint of a stream scoring, or which rate of a Transformer's")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    records, sides = [], []
    for (record, bits), row in ((args.first, args.first_row), (args.second, args.second_row)):
        records.append(json.loads(record.read_text()))
        sides.append(scoring(records[-1], np.load(bits), row))
    for key in SHARED:
        if records[0].get(key) != records[1].get(key):
            parser.error(f"the scorings differ in {key}: {records[0].get(key)} and {records[1].get(key)}")
    (first, piece, targets, name), (second, other_piece, other_targets, other_name) = sides
    if targets != other_targets:
        parser.error(f"the scorings count {targets:,} and {other_targets:,} targets")
    longer = max(piece, other_piece)
    if longer % piece or longer % other_piece:
        parser.error(f"pieces of {piece} and {other_piece} bytes do not line up")
    first, second = regroup(first, longer // piece), regroup(second, longer // other_piece)
    if first.shape != second.shape:
        parser.error(f"the scorings hold {first.shape} and {second.shape} streams and pieces")
    difference, interval = compare(first, second, targets)
    result = {"split": records[0]["split"], "streams": records[0]["streams"],
              "unscored_lead_bytes": records[0]["unscored_lead_bytes"], "targets": targets,
              "piece_bytes": longer, "pieces": int(first.size),
              "first": {"scoring": name, "bpb": float(first.sum() / targets)},
              "second": {"scoring": other_name, "bpb": float(second.sum() / targets)},
              "bpb_second_over_first": difference, "bpb_second_over_first_paired_95": interval}
    if "first_bytes_of_split" in records[0]:
        result["first_bytes_of_split"] = records[0]["first_bytes_of_split"]
    print(f"{result['first']['bpb']:.6f} and {result['second']['bpb']:.6f} bpb | second over first "
          f"{difference:+.4f} [{interval[0]:+.4f}, {interval[1]:+.4f}] over {first.size:,} pieces of {longer} bytes")
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
