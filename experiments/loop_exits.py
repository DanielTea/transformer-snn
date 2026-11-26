"""How early a racing loop could decide each byte (--stack-loop-exits), scored on the first bytes of a split as streams.

A racing loop's output neurons sum the evidence of every pass of its loop: after the first pass, after the second,
..., and at the top. This reads the stack as eval_stream.py does and keeps, for every scored byte, the bits of each
pass's summed evidence and its margin (how many bits the most likely byte leads the next). A byte decided at the first
pass whose margin reaches a threshold takes that pass's bits; a byte that never reaches it takes the top's. For a grid
of thresholds this reports the mean bits and the mean number of passes used (the top counts as the loop's last pass),
beside the bits of every pass alone and of the best pass for every byte (a bound no rule reaches). Every pass is run
for every byte here, so later bytes read what the later passes wrote: the curve measures how early bytes settle, not
yet what a loop that stops saves.

usage: python experiments/loop_exits.py --checkpoint run/model.pt.averaged --enwik8 data/enwik8 --split validation \\
    --bytes 2000000 --json exits.json [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from eval_stream import stream_rows  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

THRESHOLDS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


@torch.no_grad()
def pass_bits(model, rows, piece, device):
    """For every scored byte: the bits and the margin of each pass's summed evidence, the top last, as
    [passes, bytes] arrays."""
    inputs, targets, scored = rows
    bits, margins, state = [], [], None
    for start in range(0, inputs.shape[1], piece):
        span = slice(start, start + piece)
        counts, state = model.stream(inputs[:, span].to(device), state)
        keep = scored[:, span].reshape(-1).bool()
        if not keep.any():
            continue
        target = targets[:, span].to(device).reshape(-1)
        rows_bits, rows_margin = [], []
        for evidence in (*model.exit_counts, counts):
            logits = scoring_logits(evidence, True, model.cfg.resolution).reshape(-1, evidence.shape[-1]).float()
            loss = F.cross_entropy(logits, target, reduction="none") / math.log(2)
            top = logits.topk(2, -1).values
            rows_bits.append(loss[keep.to(loss.device)].cpu())
            rows_margin.append(((top[:, 0] - top[:, 1]) / math.log(2))[keep.to(loss.device)].cpu())
        bits.append(torch.stack(rows_bits))
        margins.append(torch.stack(rows_margin))
    return torch.cat(bits, 1).double().numpy(), torch.cat(margins, 1).double().numpy()


def halting_curve(bits, margins, thresholds=THRESHOLDS):
    """The mean bits and passes when every byte is decided at the first earlier pass whose margin reaches each
    threshold, or at the top."""
    passes = bits.shape[0]
    curve = []
    for threshold in thresholds:
        reached = margins[:-1] >= threshold
        first = np.where(reached.any(0), reached.argmax(0), passes - 1)
        chosen = bits[first, np.arange(bits.shape[1])]
        curve.append({"margin_bits": threshold, "bits_per_byte": round(float(chosen.mean()), 6),
                      "mean_passes": round(float(first.mean() + 1), 4),
                      "decided_early": round(float((first < passes - 1).mean()), 4)})
    return curve


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=2_000_000)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=8192)
    parser.add_argument("--piece", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))[:args.bytes]
    rows = stream_rows(data, args.streams, args.lead)
    record = {"split": args.split, "first_bytes_of_split": args.bytes, "streams": args.streams, "lead": args.lead,
              "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if not getattr(model, "exits", ()):
            parser.error(f"{path}: not a racing loop (--stack-loop-exits)")
        bits, margins = pass_bits(model, rows, args.piece, device)
        best = bits.min(0)
        record["runs"][path.parent.name] = {
            "run_name": checkpoint.get("run_name"), "weights": path.name, "passes": bits.shape[0],
            "scored_bytes": int(bits.shape[1]),
            "bits_per_byte_by_pass": [round(float(row.mean()), 6) for row in bits],
            "best_pass_for_every_byte": round(float(best.mean()), 6),
            "halting": halting_curve(bits, margins)}
        run = record["runs"][path.parent.name]
        print(path.parent.name, "by pass", run["bits_per_byte_by_pass"], "halting",
              [(row["margin_bits"], row["bits_per_byte"], row["mean_passes"]) for row in run["halting"]], flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
