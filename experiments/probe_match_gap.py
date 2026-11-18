"""Split a decoder's gap to full attention by the longest earlier match of the bytes just read.

The losses are the bits per target that probe_state_attention.py writes for
its test windows. For every target, the match is the longest run of bytes
ending at the last byte read that also ends at an earlier byte of the window,
where the byte after that earlier run lies within the reach. Among equally
long matches the most recent counts. Its follower is the byte after it. A
lookup of what followed can only help directly where the follower is the
target, so those targets are called copyable. The record gives the share of
the gap and of every arm's gain that falls on each kind of target, with
intervals that resample windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import _atomic_output

HISTORIES = ((9, 32), (33, 96), (97, 512), (33, 512))
MATCHES = ((1, 1), (2, 3), (4, 7), (8, 15), (16, None))


def earlier_matches(tokens, reach):
    """Length of the longest earlier match before every target, and whether its follower is the target."""
    windows, steps = tokens.shape[0], tokens.shape[1] - 1
    length, copyable = np.zeros((windows, steps), np.int16), np.zeros((windows, steps), bool)
    previous, rows = np.zeros((windows, steps), np.int16), np.arange(windows)
    for last in range(1, steps):
        # The run ending at an earlier byte grows by one where that byte equals the last byte read.
        grown = np.zeros((windows, last), np.int16)
        grown[:, 1:] = previous[:, :last - 1] + 1
        grown[:, 0] = 1
        previous[:, :last] = np.where(tokens[:, :last] == tokens[:, last:last + 1], grown, 0)
        # The follower of the earlier byte s is byte s + 1, which lies last - s - 1 bytes back.
        reachable = previous[:, max(0, last - reach):last]
        longest = reachable.max(axis=1)
        earlier = last - 1 - np.argmax(reachable[:, ::-1] == longest[:, None], axis=1)
        length[:, last] = longest
        copyable[:, last] = (longest > 0) & (tokens[rows, earlier + 1] == tokens[:, last + 1])
    return length, copyable


def share(part, whole, resamples):
    """A ratio of two sums over windows, with the 95% interval of resampled windows."""
    return {"share": float(part.sum() / whole.sum()),
            "paired_95": np.percentile(part[resamples].sum(axis=1) / whole[resamples].sum(axis=1), (2.5, 97.5)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True, help="the recorded test windows, [windows, bytes + 1]")
    parser.add_argument("--losses", type=Path, required=True, nargs="+",
                        help="bits per target as .npz files, later files add the arms the earlier ones lack")
    parser.add_argument("--reach", type=int, default=128, help="bytes back that a follower may lie")
    parser.add_argument("--decoder", default="decoder", help="the frozen decoder's entry")
    parser.add_argument("--reference", default="reference", help="the Transformer's entry")
    parser.add_argument("--control", default="own", help="the arm that every other arm's gain is measured against")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.reach < 1:
        parser.error("the reach must be positive")
    if args.json.exists():
        parser.error(f"{args.json} already exists")

    tokens = np.load(args.tokens).astype(np.int16)
    losses = {}
    for path in args.losses:
        with np.load(path) as stored:
            losses.update({name: stored[name].astype(np.float64) for name in stored.files if name not in losses})
    windows, steps = tokens.shape[0], tokens.shape[1] - 1
    if any(name not in losses for name in (args.decoder, args.reference, args.control)):
        parser.error("the losses lack the decoder, the reference or the control")
    if any(loss.shape != (windows, steps) for loss in losses.values()):
        parser.error("the losses do not belong to these windows")
    arms = [name for name in losses if name not in (args.decoder, args.reference, args.control)]
    length, copyable = earlier_matches(tokens, args.reach)
    resamples = np.random.default_rng(5261).integers(windows, size=(3000, windows))
    gap = losses[args.decoder] - losses[args.reference]
    gains = {name: losses[args.control] - losses[name] for name in arms}
    # The two wholes come first. The finer kinds give the match length and what followed the match.
    kinds = [("copyable", copyable), ("not copyable", ~copyable), ("no match", length == 0)]
    for low, high in MATCHES:
        within = (length >= low) & (length <= (high or steps))
        label = f"{low}" if low == high else f"{low}-{high}" if high else f"{low}+"
        kinds += [(f"{label} target", within & copyable), (f"{label} other", within & ~copyable)]

    histories = []
    for low, high in HISTORIES:
        if high > steps:
            continue
        chosen = np.zeros((windows, steps), bool)
        chosen[:, low - 1:high] = True
        whole_gap = (gap * chosen).sum(axis=1)
        row = {"preceding_bytes": [low, high], "targets": int(chosen.sum()),
               "decoder_bpb": float(losses[args.decoder][chosen].mean()),
               "reference_bpb": float(losses[args.reference][chosen].mean()), "gap_bpb": float(gap[chosen].mean()),
               "gain_over_control_bpb": {name: float(gain[chosen].mean()) for name, gain in gains.items()},
               "kinds": []}
        print(f"\n{low}-{high} bytes of history | gap {row['gap_bpb']:.4f} | "
              + " | ".join(f"{name} {gain:+.4f}" for name, gain in row["gain_over_control_bpb"].items()))
        for label, kind in kinds:
            part = chosen & kind
            if not part.any():
                continue
            entry = {"kind": label, "targets_share": float(part.sum() / chosen.sum()),
                     "decoder_bpb": float(losses[args.decoder][part].mean()),
                     "control_bpb": float(losses[args.control][part].mean()),
                     "reference_bpb": float(losses[args.reference][part].mean()), "gap_bpb": float(gap[part].mean()),
                     "gap": share((gap * part).sum(axis=1), whole_gap, resamples), "arms": {}}
            for name, gain in gains.items():
                inside = (gain * part).sum(axis=1)
                entry["arms"][name] = {"gain_over_control_bpb": float(gain[part].mean()),
                                       "gain": share(inside, (gain * chosen).sum(axis=1), resamples)}
                # The gap of a fine kind can be near zero, so only the two wholes say how much of it an arm closes.
                if "copyable" in label:
                    entry["arms"][name]["gap_closed"] = share(inside, (gap * part).sum(axis=1), resamples)
            row["kinds"].append(entry)
            print(f"{label:<13} {entry['targets_share']:6.1%} of targets | decoder {entry['decoder_bpb']:.3f} | "
                  f"transformer {entry['reference_bpb']:.3f} | gap {entry['gap_bpb']:+.3f}, "
                  f"{entry['gap']['share']:5.1%} of the gap | "
                  + " | ".join(f"{name} {arm['gain_over_control_bpb']:+.3f}"
                               + (f" ({arm['gap_closed']['share']:.0%} of this gap)" if "gap_closed" in arm else "")
                               for name, arm in entry["arms"].items()))
        histories.append(row)

    result = {"tokens": str(args.tokens), "losses": [str(path) for path in args.losses], "reach": args.reach,
              "windows": windows, "window_bytes": steps, "resamples": len(resamples), "decoder": args.decoder,
              "reference": args.reference, "control": args.control, "histories": histories}
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
