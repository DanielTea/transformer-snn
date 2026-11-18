"""Which weights carry the loss that averaging removes? Swap the averaged weights in one group at a time.

usage: attribute_jitter.py PARTIAL_CHECKPOINT [--windows 64]     (a partial saved by a run with --weight-average)
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import EventLM  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--windows", type=int, default=64)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = load_checkpoint(args.checkpoint, device)
model.eval()
averaged = {name: value.to(device) for name, value in checkpoint["averaged_parameters"].items()}
event = isinstance(model, EventLM)
resolution = model.cfg.resolution if event else 1


def group_of(name):
    parts = name.split(".")
    if parts[0] == "state_reads":
        return "read " + parts[2]
    if parts[0] == "blocks":
        index = int(parts[1])
        rest = re.sub(r"\.(weight|bias)$", "", ".".join(part for part in parts[2:] if not part.isdigit()))
        band = "blocks 0-3" if index < 4 else "blocks 4-8" if index < 9 else "blocks 9+"
        return f"{band} {rest}"
    return parts[0]


parameters = dict(model.named_parameters())
raw = {name: value.detach().clone() for name, value in parameters.items()}
groups = {}
for name in parameters:
    groups.setdefault(group_of(name), []).append(name)
_, validation, _ = load_splits(Path("data/enwik8"))
data = torch.from_numpy(np.array(validation))
starts = torch.linspace(0, data.numel() - 514, args.windows).long()
windows = torch.stack([data[start:start + 513] for start in starts]).to(device=device, dtype=torch.long)


@torch.inference_mode()
def loss():
    total = 0.0
    for batch in windows.split(16):
        logits = scoring_logits(model(batch[:, :-1]), event, resolution)
        total += F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="sum").item()
    return total / (windows.shape[0] * 512) / math.log(2)


def use(names, source):
    with torch.no_grad():
        for name in names:
            parameters[name].copy_(source[name])


base = loss()
use(parameters, averaged)
mean = loss()
use(parameters, raw)
print(f"{args.checkpoint} step {checkpoint['args']['steps']} | raw {base:.4f} | averaged {mean:.4f} | "
      f"averaging removes {base - mean:.4f} bpb", flush=True)
print(f"{'group':<30} {'weights':>11} {'rms(raw-avg)':>13} {'rms/mean|w|':>12} {'avg here only':>14} {'raw here only':>14}")
for group, names in groups.items():
    count = sum(parameters[name].numel() for name in names)
    gap = math.sqrt(sum((raw[name] - averaged[name]).square().sum().item() for name in names) / count)
    size = sum(raw[name].abs().sum().item() for name in names) / count
    use(names, averaged)
    only_averaged = loss() - base
    use(parameters, averaged)
    use(names, raw)
    only_raw = loss() - mean
    use(parameters, raw)
    print(f"{group:<30} {count:>11,} {gap:13.5f} {gap / size:12.3f} {only_averaged:+14.4f} {only_raw:+14.4f}", flush=True)
