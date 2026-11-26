"""How far a sleeping stack's gates open, on the first bytes of a split as streams.

A sleeping stack (--stack-sleep WINDOW) replays every window of WINDOW bytes of a stream once the window has ended,
and every read writes the cells that its replay made back into its tape. For every stored cell of every head a gate
cell of the tags and one of the payloads fire the quarters of the stored cell that stay (0 to 4), and the cell takes
the rest from the replay: a gate that fires 4 quarters keeps the cell as the tape held it, one that fires none
replaces it. Gates that start closed (--stack-sleep-keep 0.9) fire 4 quarters until training opens them.

This runs the streams as eval_stream.py does and counts, for every read and kind of cell, how the quarters that the
gates keep spread over 0 to 4 and their mean share, over the windows replayed after the lead bytes. A read that the
loop runs several times counts all its passes together.

usage: python experiments/sleep_gates.py --checkpoint run/model.pt.averaged [more ...] --enwik8 data/enwik8 \\
    --split validation --bytes 65536 --json sleep_gates.json [--device mps]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from eval_stream import stream_rows  # noqa: E402
from snnformer.event_lm import _SLEEP_QUARTERS, EventLM  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

KINDS = ("tags", "payloads")


def kept_quarters(read, tape, tag, payload):
    """The quarters that a read's gate cells keep of every stored cell of the window just replayed, as one tensor
    [batch, heads, cells] for the tags and one for the payloads, computed as ``_StateRead._blend`` computes them."""
    steps = tag.shape[2]
    ticks, rests = read.sleep_ticks(), read.sleep_rest * read.rate
    unit = (1 << read.sleep_bits) * read.resolution
    kept = []
    for kind, (made, stored) in enumerate(((tag, tape.tag), (payload, tape.payload))):
        old = stored[:, :, -steps:]
        charge = torch.einsum("bhsc,hc->bhs", torch.cat((made, old), -1), ticks[kind])
        kept.append(((charge / unit + rests[kind][:, None]).sigmoid() * _SLEEP_QUARTERS).round())
    return kept


def sleep_gates(model, inputs, piece, device, skip):
    """For every sleeping read (by module name) and kind: how many stored cells the gates kept 0 to 4 quarters of,
    over the windows replayed after the first ``skip`` bytes of every stream."""
    reads = [(name, module) for name, module in model.named_modules() if getattr(module, "sleep", 0)
             and hasattr(module, "sleep_gate")]
    if not reads:
        raise ValueError("the stack does not sleep")
    counts = {name: np.zeros((len(KINDS), _SLEEP_QUARTERS + 1), dtype=np.int64) for name, _ in reads}
    counting = [False]
    for name, read in reads:
        # The read's own method, wrapped on the instance for this pass and removed after it.
        def blend(tape, tag, payload, pulse_counts=None, *, _name=name, _read=read, _original=read._blend):
            if counting[0]:
                for kind, kept in enumerate(kept_quarters(_read, tape, tag, payload)):
                    found = torch.bincount(kept.flatten().long(), minlength=_SLEEP_QUARTERS + 1)
                    counts[_name][kind] += found.cpu().numpy()
            return _original(tape, tag, payload, pulse_counts)

        read._blend = blend
    try:
        state = None
        with torch.no_grad():
            for start in range(0, inputs.shape[1], piece):
                counting[0] = start >= skip
                _, state = model.stream(inputs[:, start:start + piece].to(device), state)
    finally:
        for _, read in reads:
            read.__dict__.pop("_blend", None)
    found = {}
    for name, table in counts.items():
        row = {}
        for kind, label in enumerate(KINDS):
            total = int(table[kind].sum())
            share = table[kind] / max(total, 1)
            row[label] = {"cells": total, "share_by_quarters_kept": [round(float(s), 4) for s in share],
                          "mean_share_kept": round(float((share * np.arange(_SLEEP_QUARTERS + 1)).sum()
                                                         / _SLEEP_QUARTERS), 4)}
        found[name] = row
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=65536, help="the first bytes of the split")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=1024, help="bytes of every stream whose windows are not counted")
    parser.add_argument("--piece", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))[:args.bytes]
    inputs, _, _ = stream_rows(data, args.streams, args.lead)
    record = {"split": args.split, "first_bytes_of_split": args.bytes, "streams": args.streams, "lead": args.lead,
              "quarters": _SLEEP_QUARTERS, "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if not isinstance(model, EventLM) or not model.cfg.stack_sleep:
            parser.error(f"{path}: the gates are those of a sleeping stack")
        found = sleep_gates(model, inputs, args.piece, device, skip=args.lead)
        record["runs"][path.parent.name] = {"run_name": checkpoint.get("run_name"), "weights": path.name,
                                            "window": model.cfg.stack_sleep, "keep_at_start": model.cfg.stack_sleep_keep,
                                            "reads": found}
        for name, row in found.items():
            print(path.parent.name, name, " ".join(
                f"{label} keep {row[label]['mean_share_kept']:.3f} (all {row[label]['share_by_quarters_kept'][-1]:.3f},"
                f" none {row[label]['share_by_quarters_kept'][0]:.3f})" for label in KINDS), flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
