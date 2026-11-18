"""How many spikes each firing population of a decoder sends at a byte, on validation text.

usage: spike_levels.py CHECKPOINT... [--windows 8] [--averaged] [--device cpu] [--json OUT]
Run from the repository root.

Every population of the decoder turns its charge into whole spikes. A neuron that sends up to n spikes at a
byte tells one of 2n + 1 values apart, so the spike counts are the precision of what a population passes on.
For every population that fires in a forward pass this prints

  the neurons of the population and the share of them that are silent at a byte,
  the mean number of spikes of a neuron at a byte, over all neurons and over those that fire,
  the 99th percentile of that number,
  the levels: the number of evenly used values with the same entropy as a neuron's spike count,
  and the population's spikes per byte.

Populations appear in the order of their first use and under the name of their module in the model.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "experiments/read_clocks")
from words_in_view import load_weights, validation_windows  # noqa: E402

from snnformer.event_lm import _IntegrateFire  # noqa: E402

# Spike counts above this are counted with it in the histogram of a population.
LIMIT = 255


def watch(model):
    """Collect a histogram of the signed spike counts of every firing population while the model runs."""
    seen, hooks = {}, []
    for name, module in model.named_modules():
        if not isinstance(module, _IntegrateFire):
            continue

        def keep(module, inputs, output, name=name):
            counts = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(counts) or counts.dim() < 2:
                return
            values = counts.detach().round().clamp(-LIMIT, LIMIT).to(torch.int64).flatten() + LIMIT
            record = seen.setdefault(name, {"histogram": torch.zeros(2 * LIMIT + 1, dtype=torch.float64),
                                            "calls": 0, "neurons": counts.shape[-1], "spikes": 0.0, "bytes": 0})
            record["histogram"] += torch.bincount(values, minlength=2 * LIMIT + 1).double().cpu()
            record["calls"] += 1
            record["spikes"] += counts.detach().abs().double().sum().item()
            # A population with a shape of [batch, bytes, ...] fires once at every byte.
            record["bytes"] += counts.shape[0] * counts.shape[1]
        hooks.append(module.register_forward_hook(keep))
    return seen, hooks


def levels(path, tokens, averaged, device):
    model, took_average = load_weights(path, averaged, device)
    model.eval()
    seen, hooks = watch(model)
    with torch.inference_mode():
        for batch in tokens.split(4):
            model(batch[:, :-1])
    for hook in hooks:
        hook.remove()
    rows = []
    for name, record in seen.items():
        share = record["histogram"] / record["histogram"].sum()
        size = (torch.arange(2 * LIMIT + 1) - LIMIT).abs().double()
        firing = share[size > 0].sum().item()
        entropy = -(share[share > 0] * share[share > 0].log()).sum().item()
        # The magnitude below which 99% of a neuron's bytes lie.
        by_size = torch.zeros(LIMIT + 1, dtype=torch.float64).index_add_(0, size.long(), share)
        high = int((by_size.cumsum(0) < 0.99).sum().item())
        mean = (share * size).sum().item()
        rows.append({"population": name, "neurons": record["neurons"], "silent": 1 - firing, "mean_spikes": mean,
                     "mean_spikes_when_firing": mean / firing if firing else 0.0, "spikes_99th_percentile": high,
                     "levels": float(torch.tensor(entropy).exp()), "passes_per_window": record["calls"],
                     "spikes_per_byte": record["spikes"] / record["bytes"]})
    return rows, took_average


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--averaged", action="store_true", help="take the running weight average of a snapshot")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tokens = torch.from_numpy(validation_windows(args.windows)).to(device)
    records = []
    for path in args.checkpoints:
        rows, took_average = levels(path, tokens, args.averaged, device)
        print(f"{path}: {'averaged weights' if took_average else 'weights as stored'}, {args.windows} validation "
              f"windows of {tokens.shape[1] - 1} bytes")
        print(f"{'population':<44} {'neurons':>7} {'silent':>7} {'spikes':>7} {'firing':>7} {'99%':>5} "
              f"{'levels':>7} {'per byte':>10}")
        for row in rows:
            print(f"{row['population']:<44} {row['neurons']:>7} {row['silent']:>7.3f} {row['mean_spikes']:>7.2f} "
                  f"{row['mean_spikes_when_firing']:>7.2f} {row['spikes_99th_percentile']:>5} {row['levels']:>7.2f} "
                  f"{row['spikes_per_byte']:>10.0f}")
        records.append({"checkpoint": path, "averaged": took_average, "windows": args.windows, "populations": rows})
    if args.json is not None:
        args.json.write_text(json.dumps({"checkpoints": records}, indent=1) + "\n")


if __name__ == "__main__":
    main()
