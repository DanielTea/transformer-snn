"""How densely the threshold cells of a stack fire, level by level, on the first bytes of a split as unbroken streams.

For every level it reports the share of cells that fire at a byte, the share of cells whose spikes make up nine
tenths of the level's spikes at a byte (as unit_density.py reports it for a Transformer's units), the share of bytes at
which each cell fires (its median and spread over the cells, and how many cells never fire), and how many spikes a
firing cell sends. For plain cells it also reports how far from its threshold a cell's charge sits: the share of
cells and bytes with a charge of so many thresholds, where a cell fires from half a threshold on.
A sparse code, as the granule cells of the cerebellum keep under the inhibition of their Golgi cells, has a small
share of cells firing at every byte and every cell firing at a small share of bytes. Streams run as
eval_stream.py runs them (16 parts side by side, the parts after the first hearing a lead of unscored bytes).

usage: cell_activity.py --checkpoint run/model.pt.averaged [more ...] --enwik8 data/enwik8 --split validation \\
    --bytes 65536 --json activity.json
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
from snnformer.event_lm import EventLM, _ExcitatoryFire, _GranuleLayer  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402
from unit_density import units_for  # noqa: E402

# The charge of a plain cell in thresholds, after its rest: the bounds of the bins and their names. A cell fires its
# first spike at half a threshold.
_CHARGE_BOUNDS = (-8.0, -4.0, -2.0, -1.0, 0.0, 0.5)
_CHARGE_BINS = ("below -8", "-8 to -4", "-4 to -2", "-2 to -1", "-1 to 0", "0 to 0.5", "0.5 and above")


def activity(model, inputs, piece, device, skip=0):
    """Per level: the shares and counts of firing over every byte of ``inputs`` after the first ``skip``."""
    layers = [module for module in model.modules() if isinstance(module, _GranuleLayer)]
    tallies = [{"bytes": 0, "firing": 0, "spikes": 0.0, "needed": 0, "per_cell": None, "counts": []} for _ in layers]
    position = [0]

    def hook(tally):
        def record(_module, _inputs, counts):
            kept = counts.detach()[:, max(0, skip - position[0]):]
            if kept.shape[1] == 0:
                return
            flat = kept.flatten(0, 1).flatten(1).float()
            firing = flat > 0
            tally["bytes"] += flat.shape[0]
            tally["firing"] += int(firing.sum())
            tally["spikes"] += float(flat.sum())
            # The cells whose largest counts add up to nine tenths of the level's spikes at a byte, counted whole so
            # that the pieces do not change the sum; a byte at which no cell fires takes none.
            needed = (units_for(flat) * flat.shape[1]).round().long() * (flat.sum(-1) > 0)
            tally["needed"] += int(needed.sum())
            per = firing.sum(0).double().cpu()
            tally["per_cell"] = per if tally["per_cell"] is None else tally["per_cell"] + per
            # The counts of firing cells are kept for their median, at most about 50,000 of each piece, evenly spread.
            fired = flat[firing]
            tally["counts"].append(fired[::max(1, fired.numel() // 50000)].cpu())
        return record

    def charges(tally):
        def record(fire, args, _kwargs):
            start = max(0, skip - position[0])
            charge = args[0].detach()[:, start:]
            if charge.shape[1] == 0:
                return
            threshold = args[1][:, start:].to(torch.float64) * (fire.threshold // fire.quanta_per_threshold)
            # A neuron counts whole units of charge, so the bins are those of the whole units.
            share = (charge.trunc().to(torch.float64) / threshold).flatten()
            bins = torch.bucketize(share, torch.tensor(_CHARGE_BOUNDS, dtype=torch.float64, device=share.device),
                                   right=True)
            held = torch.bincount(bins, minlength=len(_CHARGE_BINS)).cpu()
            tally["charge"] = held if tally.get("charge") is None else tally["charge"] + held
        return record

    handles = [layer.fire.register_forward_hook(hook(tally)) for layer, tally in zip(layers, tallies)]
    # Plain cells alone: a gated, vetoed, shaped or racing cell does not fire by its own charge alone.
    handles += [layer.fire.register_forward_pre_hook(charges(tally), with_kwargs=True)
                for layer, tally in zip(layers, tallies) if type(layer.fire) is _ExcitatoryFire]
    try:
        state = None
        with torch.no_grad():
            for start in range(0, inputs.shape[1], piece):
                _, state = model.stream(inputs[:, start:start + piece].to(device), state)
                position[0] = start + piece
    finally:
        for handle in handles:
            handle.remove()
    levels = []
    for tally in tallies:
        cells = tally["per_cell"].numel()
        share = (tally["per_cell"] / tally["bytes"]).numpy()
        counts = torch.cat(tally["counts"]).numpy() if tally["counts"] else np.zeros(1)
        by_charge = {}
        if tally.get("charge") is not None:
            by_charge = {"share_of_cells_and_bytes_by_charge_in_thresholds": {
                name: int(held) / (tally["bytes"] * cells) for name, held in zip(_CHARGE_BINS, tally["charge"])}}
        levels.append({
            "cells": cells,
            "share_of_cells_firing_at_a_byte": tally["firing"] / (tally["bytes"] * cells),
            "share_of_cells_sending_nine_tenths": tally["needed"] / (tally["bytes"] * cells),
            "share_of_bytes_a_cell_fires_at": {"median": float(np.median(share)), "p10": float(np.quantile(share, 0.1)),
                                               "p90": float(np.quantile(share, 0.9))},
            "cells_that_never_fire": int((share == 0).sum()),
            "spikes_of_a_firing_cell": {"median": float(np.median(counts)), "p90": float(np.quantile(counts, 0.9))},
            "cell_spikes_per_byte": tally["spikes"] / tally["bytes"],
            **by_charge,
        })
    return levels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=65536, help="the first bytes of the split")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=1024, help="unscored bytes that a part after the first hears")
    parser.add_argument("--piece", type=int, default=512)
    parser.add_argument("--averaged", action="store_true", help="use the checkpoint's averaged weights")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))[:args.bytes]
    inputs, _, _ = stream_rows(data, args.streams, args.lead)
    record = {"split": args.split, "first_bytes_of_split": args.bytes, "streams": args.streams, "lead": args.lead,
              "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if not isinstance(model, EventLM) or not model.cfg.stack:
            parser.error(f"{path}: cell activity is measured on a stack of reads and cells")
        if args.averaged:
            model.load_state_dict(checkpoint["averaged_parameters"], strict=False)
            model.synaptic_scaling()
        name = checkpoint.get("run_name", path.parent.name)
        levels = activity(model, inputs, args.piece, device, skip=args.lead)
        record["runs"][str(path.parent.name)] = {"run_name": name, "updates": checkpoint["args"]["steps"],
                                                 "levels": levels}
        print(path.parent.name, " ".join(f"{level['share_of_cells_firing_at_a_byte']:.3f}" for level in levels),
              "| nine tenths from", " ".join(f"{level['share_of_cells_sending_nine_tenths']:.3f}" for level in levels),
              "| median spikes", " ".join(f"{level['spikes_of_a_firing_cell']['median']:.0f}" for level in levels),
              "| never", sum(level["cells_that_never_fire"] for level in levels), flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
