"""How many of a Transformer reference's feed-forward units are open at a byte, block by block.

cell_activity.py measures the stack's threshold cells: the share of cells that fire at a byte. This asks the same
of the Transformer reference. A gated unit is open where the drive of its gate is above zero, a plain unit where
its own drive is. For every block it reports the share of units open at a byte, the share of bytes at which each
unit is open (its median and spread over the units), and the share of units that carry nine tenths of what the
block's units pass on at a byte. The Transformer reads windows. Only the second half of every window is counted,
so that every counted byte has at least half a window of context.

usage: unit_density.py --checkpoint run/model.pt.averaged [more ...] --enwik8 data/enwik8 --split validation \\
    --bytes 65536 --window 256 --json density.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from snnformer.event_lm import EventLM  # noqa: E402
from snnformer.transformer_reference import _TransformerBlock  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402


def passed_on(drive, plain, smooth):
    """Which units are open, and what each passes on, from the drive of a block's units as [..., units or 2 units]."""
    if plain:
        return drive > 0, F.gelu(drive) if smooth else F.relu(drive)
    gate, value = drive.chunk(2, dim=-1)
    return gate > 0, (F.silu(gate) if smooth else F.relu(gate)) * value


def units_for(passed, share=0.9):
    """Per byte, the share of units whose largest outputs add up to ``share`` of what all units pass on."""
    size = passed.abs().sort(dim=-1, descending=True).values
    total = size.sum(-1, keepdim=True).clamp_min(torch.finfo(size.dtype).tiny)
    # The units strictly below the share, plus the one that reaches it.
    needed = (size.cumsum(-1) / total < share).sum(-1) + 1
    return needed.clamp(max=size.shape[-1]).to(size.dtype) / size.shape[-1]


def density(model, windows, batch=16):
    """Per block: how many units are open, over the second half of every window of ``windows`` [windows, bytes]."""
    blocks = [module for module in model.modules() if isinstance(module, _TransformerBlock)]
    tallies = [{"bytes": 0, "open": 0.0, "per_unit": None, "needed": 0.0} for _ in blocks]

    def hook(block, tally):
        def record(_module, _inputs, drive):
            is_open, passed = passed_on(drive.detach()[:, drive.shape[1] // 2:], block.plain, block.smooth)
            is_open = is_open.reshape(-1, is_open.shape[-1]).to(drive.dtype)
            tally["bytes"] += is_open.shape[0]
            tally["open"] += is_open.sum().item()
            per_unit = is_open.sum(0).cpu()
            tally["per_unit"] = per_unit if tally["per_unit"] is None else tally["per_unit"] + per_unit
            tally["needed"] += units_for(passed.reshape(-1, passed.shape[-1])).sum().item()
        return record

    handles = [block.mlp_in.register_forward_hook(hook(block, tally)) for block, tally in zip(blocks, tallies)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, windows.shape[0], batch):
                model(windows[start:start + batch])
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    rows = []
    for block, tally in zip(blocks, tallies):
        units = tally["per_unit"].numel()
        share = (tally["per_unit"] / tally["bytes"]).numpy()
        rows.append({
            "units": units,
            "unit": ("plain" if block.plain else "gated") + (", smooth" if block.smooth else ", hard"),
            "share_of_units_open_at_a_byte": tally["open"] / (tally["bytes"] * units),
            "share_of_bytes_a_unit_is_open_at": {"median": float(np.median(share)), "p10": float(np.quantile(share, 0.1)),
                                                 "p90": float(np.quantile(share, 0.9))},
            "share_of_units_passing_nine_tenths": tally["needed"] / tally["bytes"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=65536, help="the first bytes of the split")
    parser.add_argument("--window", type=int, default=256, help="bytes of a window; its second half is counted")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.window, args.batch) < 2 or args.bytes < args.window:
        parser.error("a window holds at least two bytes, a pass at least two windows, and the bytes at least a window")
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    count = args.bytes // args.window
    data = torch.from_numpy(np.array((test if args.split == "test" else validation)[:count * args.window]))
    windows = data.long().view(count, args.window).to(device)
    record = {"split": args.split, "first_bytes_of_split": count * args.window, "window_bytes": args.window,
              "counted_bytes": count * (args.window - args.window // 2), "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if isinstance(model, EventLM):
            parser.error(f"{path}: cell_activity.py measures a stack; this measures the Transformer reference")
        blocks = density(model, windows, args.batch)
        record["runs"][str(path.parent.name)] = {"run_name": checkpoint.get("run_name", path.parent.name),
                                                 "updates": checkpoint["args"]["steps"], "blocks": blocks}
        print(path.parent.name, " ".join(f"{block['share_of_units_open_at_a_byte']:.3f}" for block in blocks),
              "| nine tenths from", " ".join(f"{block['share_of_units_passing_nine_tenths']:.3f}" for block in blocks),
              flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
