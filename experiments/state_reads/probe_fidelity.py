"""What does one step along the task gradient buy, on the bytes it was taken from and on other bytes?

usage: probe_fidelity.py [--averaged] CHECKPOINT...   (run from the repository root)
For every checkpoint: the agreement of the task gradients of eight training-sized batches, and the change of the
loss after one step along the gradient of those 64 windows, measured on the same windows and on 64 others.
The step moves every weight tensor by ETA times its gradient divided by the gradient's root mean square, so a
weight moves about as far as an optimizer step of that learning rate would move it.
"""
import argparse
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
parser.add_argument("checkpoints", type=Path, nargs="+")
parser.add_argument("--averaged", action="store_true")
parser.add_argument("--split", default="validation", choices=("train", "validation"))
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(train if args.split == "train" else validation)
starts = np.linspace(0, len(data) - 514, 128).astype(np.int64)
windows = torch.from_numpy(np.stack([data[start:start + 513] for start in starts]).astype(np.int64)).to(device)
taken, others = windows[0::2], windows[1::2]
etas = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3)

for path in args.checkpoints:
    model, checkpoint = load_checkpoint(path, device)
    averaged = checkpoint.get("averaged_parameters") if args.averaged else None
    if averaged is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaged[name].to(parameter))
    event = isinstance(model, EventLM)
    resolution = model.cfg.resolution if event else 1
    model.train(event)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    def loss_of(batch):
        logits = scoring_logits(model(batch[:, :-1]), event, resolution)  # noqa: F821
        return F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten())

    def mean_loss(rows):
        with torch.no_grad():
            return torch.stack([loss_of(batch) for batch in rows.split(8)]).mean().item()

    gradients = []
    for batch in taken.split(8):
        grads = torch.autograd.grad(loss_of(batch), parameters, allow_unused=True)
        gradients.append([torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads, strict=True)])
    flat = torch.stack([torch.cat([g.flatten() for g in grads]) for grads in gradients])
    unit = F.normalize(flat, dim=1)
    cosine = unit @ unit.T
    agreement = (cosine.sum() - cosine.diagonal().sum()).item() / (len(flat) * (len(flat) - 1))
    mean = [torch.stack(parts).mean(0) for parts in zip(*gradients, strict=True)]
    del flat, unit, gradients
    direction = [g / g.square().mean().sqrt().clamp_min(1e-30) for g in mean]
    slope = sum((g * d).sum().item() for g, d in zip(mean, direction, strict=True))
    before = mean_loss(taken), mean_loss(others)
    again = mean_loss(taken)
    saved = [parameter.detach().clone() for parameter in parameters]
    print(f"{path} | steps {checkpoint['args']['steps']} | {'averaged' if averaged is not None else 'raw'} | "
          f"loss {before[0]:.4f} taken, {before[1]:.4f} others (repeat differs by {again - before[0]:+.1e}) | "
          f"agreement of batch gradients {agreement:.3f}", flush=True)
    for eta in etas:
        with torch.no_grad():
            for parameter, start, d in zip(parameters, saved, direction, strict=True):
                parameter.copy_(start - eta * d)
        after = mean_loss(taken), mean_loss(others)
        print(f"  step {eta:.0e}: predicted {-eta * slope:+.5f} nats | taken {after[0] - before[0]:+.5f} "
              f"({(after[0] - before[0]) / (-eta * slope):5.2f} of predicted) | others {after[1] - before[1]:+.5f} "
              f"({(after[1] - before[1]) / (-eta * slope):5.2f})", flush=True)
    with torch.no_grad():
        for parameter, start in zip(parameters, saved, strict=True):
            parameter.copy_(start)
    del model, saved, direction, mean
    torch.cuda.empty_cache()
