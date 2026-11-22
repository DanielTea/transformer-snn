"""Where one scoring of a split beats another: the paired bits of two scorings of the same streams, split by what
the scored bytes are.

eval_stream.py (``--bits``) and eval_taught_reference.py keep the bits of every piece of every stream. The two
scorings are summed to common pieces, and every target is given a kind twice over:

* what it is: a letter, a digit, a space, a line break, wiki or XML markup, other punctuation, or a byte of a
  multi-byte character;
* what its context offers: the eight bytes before it ran earlier in its stream, at most ``--near`` bytes back,
  and the byte that followed them then is the target (copyable near); the same, but only from ``--near`` to
  ``--far`` bytes back (copyable far only); they ran earlier and something else followed (seen, other follower);
  or they did not run in the last ``--far`` bytes (new).

A piece holds a mix of kinds, so the gap of each kind is fitted as the least-squares solution of
``gap(piece) = sum over kinds of bytes(piece, kind) * gap(kind)`` over all pieces, with a 95% interval from
resampled pieces. The gaps of the kinds, weighted by their bytes, add up to the whole gap.

usage: gap_by_piece.py --enwik8 data/enwik8 --split test --first tf.json tf.npy --first-row 0 \\
    --second stack.json stack.npy --json gap.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import _atomic_output
from eval_stream import stream_rows
from train_lm import load_splits

import torch

LETTERS = set(range(ord("a"), ord("z") + 1)) | set(range(ord("A"), ord("Z") + 1))
DIGITS = set(range(ord("0"), ord("9") + 1))
MARKUP = set(b"[]{}|<>=&#*")
WHAT = ("letter", "digit", "space", "line break", "markup", "punctuation", "multi-byte")
CONTEXT = ("copyable near", "copyable far only", "seen, other follower", "new")
GRAM = 8


def byte_kinds(data):
    """The kind of every byte of ``data`` (uint8), as an index into WHAT."""
    table = np.full(256, WHAT.index("punctuation"), np.int8)
    for value in LETTERS:
        table[value] = WHAT.index("letter")
    for value in DIGITS:
        table[value] = WHAT.index("digit")
    for value in MARKUP:
        table[value] = WHAT.index("markup")
    table[ord(" ")] = WHAT.index("space")
    table[ord("\n")] = WHAT.index("line break")
    table[128:] = WHAT.index("multi-byte")
    return table[data]


def context_kinds(data, near, far):
    """For every position t of ``data`` (uint8), what its context offers, as an index into CONTEXT: the latest
    earlier position v with the same eight bytes before it, how far back it lies, and whether data[v] == data[t]."""
    n = len(data)
    kinds = np.full(n, CONTEXT.index("new"), np.int8)
    # The eight bytes before t as one 64-bit number, for every t >= GRAM.
    grams = np.zeros(n, np.uint64)
    for offset in range(1, GRAM + 1):
        grams[GRAM:] = (grams[GRAM:] << np.uint64(8)) | data[GRAM - offset:n - offset].astype(np.uint64)
    positions = np.arange(GRAM, n)
    order = positions[np.lexsort((positions, grams[GRAM:]))]
    same = grams[order[1:]] == grams[order[:-1]]
    later, earlier = order[1:][same], order[:-1][same]
    distance = later - earlier
    copied = data[earlier] == data[later]
    kinds[later[distance <= far]] = CONTEXT.index("seen, other follower")
    kinds[later[(distance <= far) & copied]] = CONTEXT.index("copyable far only")
    kinds[later[(distance <= near) & copied]] = CONTEXT.index("copyable near")
    return kinds


def piece_bits(bits, piece, common):
    """Per-piece bits [streams, pieces] summed to pieces of ``common`` bytes."""
    if common % piece:
        raise ValueError("the common piece must be a whole number of a scoring's pieces")
    group = common // piece
    streams, pieces = bits.shape
    padded = np.zeros((streams, -(-pieces // group) * group))
    padded[:, :pieces] = bits
    return padded.reshape(streams, -1, group).sum(2)


def fit(counts, gaps, resamples):
    """Least-squares gaps per kind from per-piece byte counts [pieces, kinds] and per-piece gap bits, with the 95%
    interval of resampled pieces; kinds without bytes get none."""
    used = counts.sum(0) > 0
    solution = np.full(counts.shape[1], np.nan)
    solution[used] = np.linalg.lstsq(counts[:, used], gaps, rcond=None)[0]
    draws = []
    for rows in resamples:
        estimate = np.full(counts.shape[1], np.nan)
        estimate[used] = np.linalg.lstsq(counts[rows][:, used], gaps[rows], rcond=None)[0]
        draws.append(estimate)
    low, high = np.nanpercentile(np.array(draws), (2.5, 97.5), axis=0)
    return solution, low, high


def table(names, counts, gaps, first, second, resamples):
    total = counts.sum()
    solution, low, high = fit(counts, gaps, resamples)
    rows = {}
    for index, name in enumerate(names):
        share = float(counts[:, index].sum() / total)
        rows[name] = {"share_of_bytes": round(share, 4),
                      "gap_bits_per_byte": None if np.isnan(solution[index]) else round(float(solution[index]), 4),
                      "gap_95": None if np.isnan(low[index]) else [round(float(low[index]), 4), round(float(high[index]), 4)],
                      "share_of_whole_gap": None if np.isnan(solution[index]) else
                      round(float(share * solution[index] / (gaps.sum() / total)), 3)}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--first", type=Path, nargs=2, required=True, metavar=("JSON", "NPY"))
    parser.add_argument("--first-row", type=int, default=0)
    parser.add_argument("--second", type=Path, nargs=2, required=True, metavar=("JSON", "NPY"))
    parser.add_argument("--second-row", type=int, default=0)
    parser.add_argument("--near", type=int, default=2032, help="bytes back that both models reach")
    parser.add_argument("--far", type=int, default=8192, help="bytes back that only the second reaches")
    parser.add_argument("--resamples", type=int, default=200)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path, _ in (args.first, args.second)]
    bits = [np.load(path)[row] for (_, path), row in ((args.first, args.first_row), (args.second, args.second_row))]
    if any(record["split"] != args.split for record in records):
        parser.error("both scorings must be of --split")
    streams, lead = records[0]["streams"], records[0]["unscored_lead_bytes"]
    if (records[1]["streams"], records[1]["unscored_lead_bytes"]) != (streams, lead):
        parser.error("the scorings cut the split into different streams")
    pieces = [record.get("piece_bytes", record.get("context_bytes")) for record in records]
    common = max(pieces)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))
    inputs, targets, scored = stream_rows(data, streams, lead)
    if lead % common:
        parser.error("the lead must be a whole number of common pieces")
    summed = [piece_bits(b, p, common) for b, p in zip(bits, pieces)]
    if summed[0].shape != summed[1].shape:
        parser.error(f"the scorings have different pieces: {summed[0].shape} and {summed[1].shape}")
    raw = data.numpy().astype(np.uint8)
    what = byte_kinds(raw)
    offered = context_kinds(raw, args.near, args.far)
    # The kinds of every target: the target at place p of a stream is byte p + 1 of the split (as stream_rows cuts).
    part = -(-(len(raw) - 1 - lead) // streams)
    places = np.arange(streams)[:, None] * part + np.arange(lead + part)
    mask = scored.numpy()
    targets_at = np.clip(places + 1, 0, len(raw) - 1)
    width = summed[0].shape[1] * common
    padded_mask = np.zeros((streams, width), bool)
    padded_mask[:, :mask.shape[1]] = mask
    padded_targets = np.zeros((streams, width), np.int64)
    padded_targets[:, :mask.shape[1]] = targets_at
    keep = padded_mask.reshape(streams, -1, common)
    target_index = padded_targets.reshape(streams, -1, common)
    counts_what = np.stack([(keep & (what[target_index] == k)).sum(2) for k in range(len(WHAT))], -1)
    counts_context = np.stack([(keep & (offered[target_index] == k)).sum(2) for k in range(len(CONTEXT))], -1)
    # Only pieces where every target is scored and has its stream's lead behind it are fitted.
    whole = keep.all(2)
    whole[0, :lead // common] = False
    gaps = (summed[1] - summed[0])[whole]
    counts_what, counts_context = counts_what[whole], counts_context[whole]
    bytes_fitted = int(counts_what.sum())
    resamples = np.random.default_rng(5261).integers(len(gaps), size=(args.resamples, len(gaps)))
    first_bpb = float(summed[0][whole].sum() / bytes_fitted)
    second_bpb = float(summed[1][whole].sum() / bytes_fitted)
    record = {"split": args.split, "streams": streams, "unscored_lead_bytes": lead, "common_piece_bytes": common,
              "pieces_fitted": int(whole.sum()), "bytes_fitted": bytes_fitted,
              "first": {"scoring": args.first[0].name, "row": args.first_row, "bpb": round(first_bpb, 5)},
              "second": {"scoring": args.second[0].name, "row": args.second_row, "bpb": round(second_bpb, 5)},
              "gap": round(second_bpb - first_bpb, 5), "gram_bytes": GRAM, "near_bytes": args.near,
              "far_bytes": args.far,
              "by_kind_of_byte": table(WHAT, counts_what, gaps, summed[0], summed[1], resamples),
              "by_what_the_context_offers": table(CONTEXT, counts_context, gaps, summed[0], summed[1], resamples)}
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=1)
        stream.write("\n")
    print(f"{bytes_fitted:,} bytes in {int(whole.sum()):,} pieces: first {first_bpb:.4f}, second {second_bpb:.4f}, "
          f"gap {second_bpb - first_bpb:+.4f}")
    for title, rows in (("kind of byte", record["by_kind_of_byte"]),
                        ("what the context offers", record["by_what_the_context_offers"])):
        print(f"by {title}:")
        for name, row in rows.items():
            print(f"  {name:22s} {row['share_of_bytes']:7.2%} of bytes  gap {row['gap_bits_per_byte']:+.4f} "
                  f"{row['gap_95']}  {row['share_of_whole_gap']:+.2f} of the whole gap")


if __name__ == "__main__":
    main()
