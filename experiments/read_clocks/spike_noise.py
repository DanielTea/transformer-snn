"""What one more rounding's worth of spike noise costs a trained decoder, on validation text.

usage: spike_noise.py CHECKPOINT... [--windows 96] [--averaged] [--scales 1 2 4] [--device cpu] [--json OUT]
Run from the repository root.

Every firing population rounds its charge to whole spikes. That leaves an error of up to half a spike, with a
variance of a twelfth of a squared spike. This adds to the spikes of a group of populations a noise of whole
spikes with that variance times a scale: one spike more or one fewer, each with a chance of scale / 24. The
loss that the noise adds says how much the score depends on the precision of those spikes. Where a noise of
the size of the rounding costs little, a finer spike resolution has little to give.

  stream        every population that writes into the stream: the input, the blocks, the cell layers, the reads
  relays        the relay neurons of the blocks
  steer         the steering neurons of the blocks
  read packets  the probes, tags and payloads of the state reads

The cells of a race are left out: their spikes are shares of a budget and no rounded charges.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments/read_clocks")
from words_in_view import load_weights, validation_windows  # noqa: E402

from snnformer.event_lm import _IntegrateFire  # noqa: E402
from train_event_lm import scoring_logits  # noqa: E402

GROUPS = {
    "stream": lambda name: name == "input_emit" or (name.endswith(".emit") and name.count(".") == 2),
    "relays": lambda name: name.endswith(".relay"),
    "steer": lambda name: name.endswith(".steer"),
    "read packets": lambda name: name.startswith("state_reads.") and name.endswith(".fire"),
}


def bits(model, tokens):
    """Mean bits per byte of the model on the windows."""
    total = 0.0
    with torch.inference_mode():
        for batch in tokens.split(4):
            logits = scoring_logits(model(batch[:, :-1]), True, model.cfg.resolution)
            total += F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="sum").item()
    return total / (tokens.shape[0] * (tokens.shape[1] - 1)) / math.log(2)


def noise_costs(path, tokens, averaged, scales, device):
    model, took_average = load_weights(path, averaged, device)
    model.eval()
    state = {"scale": 0.0, "group": None, "generator": torch.Generator()}

    def noisy(output, name):
        group = state["group"]
        if group is None or not GROUPS[group](name) or not torch.is_tensor(output) or output.dim() < 2:
            return None
        chance = state["scale"] / 24
        draw = torch.rand(output.shape, generator=state["generator"]).to(output.device)
        return output + (draw < chance).to(output.dtype) - (draw > 1 - chance).to(output.dtype)

    populations = {group: [] for group in GROUPS}
    for name, module in model.named_modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda module, inputs, output, name=name: noisy(output, name))
            for group, belongs in GROUPS.items():
                if belongs(name):
                    populations[group].append(name)
    base = bits(model, tokens)
    rows = []
    for group in GROUPS:
        added = []
        for scale in scales:
            state.update(scale=scale, group=group)
            # Every setting draws the same noise, so that the scales differ in nothing else.
            state["generator"].manual_seed(0)
            added.append(bits(model, tokens) - base)
        rows.append({"noise_on": group, "populations": populations[group], "added_bpb": added})
    return base, rows, took_average


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--windows", type=int, default=96)
    parser.add_argument("--averaged", action="store_true", help="take the running weight average of a snapshot")
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    tokens = torch.from_numpy(validation_windows(args.windows)).to(device)
    records = []
    for path in args.checkpoints:
        base, rows, took_average = noise_costs(path, tokens, args.averaged, args.scales, device)
        print(f"{path}: {'averaged weights' if took_average else 'weights as stored'}, {args.windows} validation "
              f"windows of {tokens.shape[1] - 1} bytes, as trained {base:.4f} bits per byte")
        print(f"{'noise on':<14} {'populations':>11} " + " ".join(f"{'x' + format(scale, 'g'):>9}" for scale in args.scales))
        for row in rows:
            print(f"{row['noise_on']:<14} {len(row['populations']):>11} "
                  + " ".join(f"{value:>+9.4f}" for value in row["added_bpb"]), flush=True)
        records.append({"checkpoint": path, "averaged": took_average, "windows": args.windows, "bpb": base,
                        "scales": args.scales, "rows": rows})
    if args.json is not None:
        args.json.write_text(json.dumps({"checkpoints": records}, indent=1) + "\n")


if __name__ == "__main__":
    main()
