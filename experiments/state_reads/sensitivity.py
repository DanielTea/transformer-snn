"""How much does the validation loss move when the weights take one optimizer-sized step in a random direction?

usage: sensitivity.py CHECKPOINT [--windows 64] [--sizes 3e-4,1.2e-3] [--seeds 2] [--depth 2]
Run from the repository root. Groups parameters by the first name components.
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
parser.add_argument("--sizes", default="3e-4,1.2e-3")
parser.add_argument("--seeds", type=int, default=2)
parser.add_argument("--list", action="store_true")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = load_checkpoint(args.checkpoint, device)
model.eval()
event = isinstance(model, EventLM)
resolution = model.cfg.resolution if event else 1


def group_of(name):
    parts = name.split(".")
    if parts[0] == "state_reads":
        return "read " + parts[2]
    if parts[0] == "blocks":
        index = int(parts[1])
        rest = ".".join(part for part in parts[2:] if not part.isdigit())
        rest = re.sub(r"\.(weight|bias)$", "", rest)
        band = "blocks 0-3" if index < 4 else "blocks 4-8" if index < 9 else "blocks 9+"
        return f"{band} {rest}" if event else f"blocks {rest}"
    return parts[0]


groups = {}
for name, parameter in model.named_parameters():
    groups.setdefault(group_of(name), []).append(parameter)
if args.list:
    for name, parameter in model.named_parameters():
        print(f"{name:<60} {tuple(parameter.shape)} mean|w| {parameter.abs().mean():.4f} -> {group_of(name)}")
    sys.exit()

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


base = loss()
print(f"{args.checkpoint} | {sum(p.numel() for p in model.parameters()):,} parameters | base {base:.4f} bpb", flush=True)
sizes = [float(size) for size in args.sizes.split(",")]
print(f"{'group':<34} {'weights':>11} {'mean|w|':>8} " + " ".join(f"{'+' + str(size):>11}" for size in sizes))
rows = dict(groups, **{"ALL": [p for ps in groups.values() for p in ps]})
for name, parameters in rows.items():
    count = sum(p.numel() for p in parameters)
    size_mean = sum(p.abs().sum().item() for p in parameters) / count
    cells = []
    for size in sizes:
        changes = []
        for seed in range(args.seeds):
            generator = torch.Generator(device=device).manual_seed(1000 + seed)
            saved = [p.detach().clone() for p in parameters]
            with torch.no_grad():
                for p in parameters:
                    p.add_((torch.randint(0, 2, p.shape, generator=generator, device=device) * 2 - 1).to(p.dtype) * size)
            changes.append(loss() - base)
            with torch.no_grad():
                for p, old in zip(parameters, saved):
                    p.copy_(old)
        cells.append(f"{np.mean(changes):+11.4f}")
    print(f"{name:<34} {count:>11,} {size_mean:8.4f} " + " ".join(cells), flush=True)
