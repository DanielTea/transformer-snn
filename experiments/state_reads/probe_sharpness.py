"""How far does the task gradient's promise hold, weight group by weight group?

usage: probe_sharpness.py [--averaged] [--etas 1e-6,1e-5,3e-5] CHECKPOINT   (run from the repository root)
The gradient of 64 windows is scaled per weight tensor to a root mean square of one. For every group of weights
the model steps ETA along it and the loss on the same windows is compared with the first-order prediction: a ratio
near 1 means the gradient tells the truth at that step size, a ratio below 0 means the step overshoots.
"""
import argparse
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
parser.add_argument("--averaged", action="store_true")
parser.add_argument("--etas", default="1e-6,1e-5,3e-5")
args = parser.parse_args()
etas = [float(eta) for eta in args.etas.split(",")]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, 128).astype(np.int64)[0::2]
windows = torch.from_numpy(np.stack([data[start:start + 513] for start in starts]).astype(np.int64)).to(device)
model, checkpoint = load_checkpoint(args.checkpoint, device)
averaged = checkpoint.get("averaged_parameters") if args.averaged else None
if averaged is not None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(averaged[name].to(parameter))
event = isinstance(model, EventLM)
resolution = model.cfg.resolution if event else 1
model.train(event)


def group_of(name):
    parts = name.split(".")
    if parts[0] == "state_reads":
        return "read " + parts[2]
    if parts[0] == "granule_layers":
        return "cells " + parts[2]
    if parts[0] == "blocks":
        index = int(parts[1])
        rest = ".".join(part for part in parts[2:] if not part.isdigit())
        rest = re.sub(r"\.(weight|bias)$", "", rest)
        band = "blocks 0-3" if index < 4 else "blocks 4-8" if index < 9 else "blocks 9+"
        return f"{band} {rest}" if event else f"blocks {rest}"
    return parts[0]


def loss_of(batch):
    logits = scoring_logits(model(batch[:, :-1]), event, resolution)
    return F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())


def mean_loss():
    with torch.no_grad():
        return torch.stack([loss_of(batch) for batch in windows.split(8)]).mean().item()


names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
total = [torch.zeros_like(parameter) for parameter in parameters]
for batch in windows.split(8):
    grads = torch.autograd.grad(loss_of(batch), parameters, allow_unused=True)
    for held, grad in zip(total, grads, strict=True):
        if grad is not None:
            held.add_(grad, alpha=8 / windows.shape[0])
direction = [g / g.square().mean().sqrt().clamp_min(1e-30) for g in total]
slopes = [(g * d).sum().item() for g, d in zip(total, direction, strict=True)]
groups = {}
for index, name in enumerate(names):
    groups.setdefault(group_of(name), []).append(index)
rows = dict(groups, ALL=list(range(len(names))))
before = mean_loss()
print(f"{args.checkpoint} | {'averaged' if averaged is not None else 'raw'} | loss {before:.4f} nats on 64 windows")
print(f"{'group':<34} {'weights':>11} {'promise/eta':>12} " + " ".join(f"{'kept at ' + format(eta, '.0e'):>14}" for eta in etas))
for name, members in rows.items():
    slope = sum(slopes[index] for index in members)
    cells = []
    for eta in etas:
        saved = [parameters[index].detach().clone() for index in members]
        with torch.no_grad():
            for index in members:
                parameters[index].sub_(eta * direction[index])
        cells.append(f"{(mean_loss() - before) / (-eta * slope):14.2f}" if slope > 0 else f"{'-':>14}")
        with torch.no_grad():
            for index, old in zip(members, saved, strict=True):
                parameters[index].copy_(old)
    count = sum(parameters[index].numel() for index in members)
    print(f"{name:<34} {count:>11,} {slope:12.1f} " + " ".join(cells), flush=True)
