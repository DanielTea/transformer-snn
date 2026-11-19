"""How the race cells of a decoder share their spikes on validation text.

usage: cell_use.py CHECKPOINT... [--windows 32] [--averaged] [--threads 8] [--json OUT]
Run from the repository root.

The cells of a cluster race for a fixed budget of spikes, as one race or in equal groups. For every cell layer
this prints, as means over the layer's races:

  the spikes of a race per byte and the cells that fire at a byte,
  the share of a byte's spikes that the busiest cell takes,
  the perplexity of a byte's spikes over the cells: the number of cells that an even split with the same
  entropy would use,
  the perplexity of the spikes of the whole text over the cells, which is the number of cells in use,
  and the shares of cells that take less than a tenth and less than a hundredth of an even share of the text.

A race in which one cell takes nearly all spikes at every byte is a lookup: the race names one of its cells
and only that cell's output synapses speak.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "experiments/read_clocks")
from words_in_view import load_weights, validation_windows  # noqa: E402

SUMS = ("active", "top", "perplexity", "spikes")


def watch(layer, races, seen):
    """Count the spikes of the layer's cells whenever the layer runs."""
    inner = layer.cells

    def cells(spikes, tokens=None):
        out = inner(spikes, tokens)
        # Spikes by batch, byte, race and cell.
        counts = out.detach().abs().double().unflatten(-1, (races, -1)).flatten(2, 3)
        share = counts / counts.sum(-1, keepdim=True).clamp_min(1)
        entropy = -(share * share.clamp_min(1e-12).log()).sum(-1)
        seen["use"] = seen.get("use", 0) + counts.sum((0, 1))
        seen["bytes"] = seen.get("bytes", 0) + counts.shape[0] * counts.shape[1]
        for name, value in (("active", (counts > 0).double().sum(-1)), ("top", share.amax(-1)),
                            ("perplexity", entropy.exp()), ("spikes", counts.sum(-1))):
            seen[name] = seen.get(name, 0) + value.sum((0, 1))
        return out

    layer.cells = cells


def cell_use(path, tokens, averaged, device):
    model, took_average = load_weights(path, averaged, device)
    model.eval()
    races = getattr(model.cfg, "granule_races", 1)
    seen = {int(index): {} for index in model.granule_layers}
    for index, layer in model.granule_layers.items():
        watch(layer, races, seen[int(index)])
    with torch.inference_mode():
        for batch in tokens.split(8):
            model(batch)
    rows = []
    for index, row in seen.items():
        share = row["use"] / row["use"].sum(-1, keepdim=True).clamp_min(1)
        cells = share.shape[-1]
        record = {"after_block": index, "races": share.shape[0], "cells_per_race": cells}
        record.update({name: float((row[name] / row["bytes"]).mean()) for name in SUMS})
        record["perplexity_over_text"] = float((-(share * share.clamp_min(1e-12).log()).sum(-1)).exp().mean())
        record["under_a_tenth"] = float((share < 0.1 / cells).double().mean())
        record["under_a_hundredth"] = float((share < 0.01 / cells).double().mean())
        rows.append(record)
    return rows, took_average


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--windows", type=int, default=32)
    parser.add_argument("--averaged", action="store_true",
                        help="read the running weight average where a checkpoint stores one")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tokens = torch.from_numpy(validation_windows(args.windows)[:, :512]).to(device)
    record = {"windows": args.windows, "averaged": args.averaged, "checkpoints": []}
    for path in args.checkpoints:
        rows, took_average = cell_use(path, tokens, args.averaged, device)
        name = f"{Path(path).parent.name}/{Path(path).name}"
        record["checkpoints"].append({"checkpoint": name, "layers": rows})
        print(f"{name} | {'averaged' if took_average else 'as stored'} | {args.windows} windows; "
              "per cell layer the mean over its races")
        print("after block  races  cells | spikes  cells firing  top share  perplexity | "
              "over the text: perplexity  under a tenth  under a hundredth")
        for row in rows:
            print(f"{row['after_block']:>11} {row['races']:>6} {row['cells_per_race']:>6} | {row['spikes']:>6.1f} "
                  f"{row['active']:>13.1f} {row['top']:>10.3f} {row['perplexity']:>11.1f} | "
                  f"{row['perplexity_over_text']:>25.1f} {row['under_a_tenth']:>14.3f} "
                  f"{row['under_a_hundredth']:>18.3f}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
