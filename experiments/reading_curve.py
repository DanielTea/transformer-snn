"""How much fast output synapses gain against how much of a stream they have read.

Two scorings of the same streams, one with the fast synapses and one without, keep their bits piece by piece
(eval_stream.py and eval_taught_reference.py with ``--bits``). The difference of the two, summed over the pieces that
follow the same amount of text read in their stream, is the gain of the synapses after that much reading. Every
stream starts its synapses at zero, and the unscored lead of a part teaches them like any other byte, so a piece at
place p of its stream follows p bytes of reading.

The pieces are binned by the kilobytes read before them, doubling from bin to bin. The byte counts are exact: the
streams are rebuilt from the data by the scorers' own rules. A series names its two bit files, the index of the
scoring in each (a stack's file holds one, a Transformer's one for every rate), the bytes of a piece and the streams:

    python experiments/reading_curve.py --json reading.json --books-dir data/pg19-test --part 262144 \\
        --series "Stack, PG-19 in parts" pg19-reach.npy:0 pg19-taught.npy:0 512
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from common import _atomic_output
from eval_stream import book_rows, stream_rows
from eval_stream_lm import load_books
from eval_warmed_windows import PG19_BOOKS
from train_lm import load_splits

# Kilobytes read before a piece: the bins double from 8 KB on, after one bin for the first 8.
FIRST = 8


def bins(top):
    """The edges of the bins in kilobytes, up to at least ``top``."""
    edges = [0, FIRST]
    while edges[-1] < top:
        edges.append(edges[-1] * 2)
    return edges


def piece_bytes(scored, piece):
    """The scored bytes of every piece of every stream as [streams, pieces]."""
    pieces = -(-scored.shape[1] // piece)
    padded = torch.zeros(scored.shape[0], pieces * piece, dtype=torch.int64)
    padded[:, :scored.shape[1]] = scored
    return padded.view(scored.shape[0], pieces, piece).sum(-1).numpy()


def curve(frozen, taught, counts, piece, edges):
    """The gain in bits per byte in each bin, pooled over the streams, and the median and quartiles of the streams'
    own gains in it, with the bytes and the streams that count there."""
    # A piece without a scored byte holds no bits in a scoring; it counts for nothing here either.
    gain = np.where(counts > 0, frozen - taught, 0.0)
    read = np.arange(gain.shape[1]) * piece / 1024
    rows = []
    for low, high in zip(edges, edges[1:]):
        inside = (read >= low) & (read < high)
        bytes_in = counts[:, inside].sum(1)
        own = gain[:, inside].sum(1)
        present = bytes_in > 0
        if not present.any():
            rows.append(None)
            continue
        per_stream = own[present] / bytes_in[present]
        rows.append({"from_kb": low, "to_kb": high, "gain": float(own.sum() / bytes_in.sum()),
                     "median": float(np.median(per_stream)), "quartiles": np.percentile(per_stream, (25, 75)).tolist(),
                     "bytes": int(bytes_in.sum()), "streams": int(present.sum())})
    return rows


def load(spec):
    """Bits of one scoring from PATH[:INDEX], as [streams, pieces]."""
    path, _, index = spec.rpartition(":") if ":" in Path(spec).name else (spec, "", "0")
    return np.load(path)[int(index)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--series", nargs=4, action="append", required=True,
                        metavar=("LABEL", "FROZEN", "TAUGHT", "PIECE"),
                        help="a label, the bits without and with the fast synapses as PATH[:INDEX], and the bytes "
                             "of a piece; every series reads the same streams")
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--enwik8", type=Path, help="canonical enwik8 file")
    dataset.add_argument("--books-dir", type=Path, help="directory of prepared PG-19 excerpts")
    parser.add_argument("--manifest", type=Path, default=PG19_BOOKS)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--part", type=int, default=262144)
    parser.add_argument("--lead", type=int, default=8192)
    args = parser.parse_args()
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    if args.enwik8 is not None:
        _, validation, test = load_splits(args.enwik8)
        data = torch.from_numpy(np.array(test if args.split == "test" else validation))
        _, _, scored = stream_rows(data, args.streams, args.lead)
        source = {"dataset": "enwik8", "split": args.split, "streams": args.streams, "unscored_lead_bytes": args.lead}
    else:
        books = load_books(args.manifest.read_bytes(), args.books_dir, 0, split=args.split)
        _, _, scored = book_rows([book.data for book in books], args.part, args.lead)
        source = {"dataset": "PG-19", "split": args.split, "manifest": args.manifest.name, "books": len(books),
                  "part_bytes": args.part, "unscored_lead_bytes": args.lead}
    edges = bins(scored.shape[1] / 1024)
    record = {**source, "edges_kb": edges, "series": []}
    for label, frozen, taught, piece in args.series:
        frozen, taught, piece = load(frozen), load(taught), int(piece)
        counts = piece_bytes(scored, piece)
        if frozen.shape != taught.shape or frozen.shape != counts.shape:
            parser.error(f"{label}: the bits do not hold these streams in pieces of {piece} bytes")
        rows = curve(frozen, taught, counts, piece, edges)
        total = float((frozen - taught).sum() / counts.sum())
        record["series"].append({"label": label, "piece_bytes": piece, "gain": total, "bins": rows})
        shown = " ".join(f"{row['gain']:+.4f}" for row in rows if row is not None)
        print(f"{label}: {total:+.4f} overall | by bin {shown}", flush=True)
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    if not math.isfinite(record["series"][0]["gain"]):
        raise SystemExit("no scored bytes")


if __name__ == "__main__":
    main()
