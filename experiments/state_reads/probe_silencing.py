"""Which part of the training signal pushes the granule cells silent.

usage: probe_silencing.py CHECKPOINT [BATCHES]   (run from the repository root)
For the task loss, the priced downstream pulses and the cells' own spikes it prints what one cell spike is worth and
how the firing changes after a step of the cells' input synapses along each averaged, Adam-normalised gradient.
"""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _IntegrateFire, emitted_pulses  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits, random_batch  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train, _, _ = load_splits(Path("data/enwik8"))
model, checkpoint = load_checkpoint(sys.argv[1], device)
batches = int(sys.argv[2]) if len(sys.argv) > 2 else 8
model.train()
cfg = model.cfg
layers = dict(model.granule_layers.items())
pulse_counts, cells = [], {}
for module in model.modules():
    if isinstance(module, _IntegrateFire):
        module.register_forward_hook(lambda module, _inputs, counts: pulse_counts.append(emitted_pulses(module, counts)))


def keep(name):
    def hook(_module, _inputs, counts):
        if counts.requires_grad:
            counts.retain_grad()
        cells[name] = counts
    return hook


for name, layer in layers.items():
    layer.fire.register_forward_hook(keep(name))


def losses(inputs, labels):
    pulse_counts.clear()
    counts, predicted, observed, *early = model(inputs, return_prediction=True, memory_pulses=pulse_counts,
                                                early_readout=cfg.clock_start)
    keys = cfg.forecast_key_channels if cfg.learn_context_keys else 0
    code = F.smooth_l1_loss(predicted[:, :-1, keys:] / cfg.resolution, observed[:, 1:, keys:].detach() / cfg.resolution)
    language = F.cross_entropy(scoring_logits(counts, True, cfg.resolution).flatten(0, 1), labels.flatten())
    early_loss = F.cross_entropy(scoring_logits(early[0], True, model.forecast_resolution).flatten(0, 1), labels.flatten())
    pulses = torch.stack(pulse_counts).sum() / inputs.numel()
    own = sum(count.abs().sum() for count in cells.values()) / inputs.numel()
    return {"task": language + 0.25 * code + 0.25 * early_loss, "language": language, "code": code, "early": early_loss,
            "pulses": pulses, "own": own}, language, pulses


generator = torch.Generator().manual_seed(11)
names = list(layers)
receives = [layers[name].receive for name in names]
first = {key: [torch.zeros_like(weight) for weight in receives]
         for key in ("task", "language", "code", "early", "pulses", "own")}
second = {key: [torch.zeros_like(weight) for weight in receives] for key in first}
worth = {key: {name: [0.0, 0.0] for name in names} for key in first}
firing = {name: 0.0 for name in names}
for _ in range(batches):
    inputs, labels = random_batch(train, 8, 512, generator)
    inputs, labels = inputs.to(device), labels.to(device)
    parts, language, pulses = losses(inputs, labels)
    kept = [cells[name] for name in names]
    for key, part in parts.items():
        gradients = torch.autograd.grad(part, receives + kept, retain_graph=True, allow_unused=True)
        for index, gradient in enumerate(gradients[:len(names)]):
            if gradient is not None:
                first[key][index] += gradient / batches
                second[key][index] += gradient.square() / batches
        for name, count, gradient in zip(names, kept, gradients[len(names):], strict=True):
            if gradient is None:
                continue
            # The parts are means over bytes, so the worth of one spike on one byte is the gradient times the bytes.
            spikes = count.detach() > 0
            worth[key][name][0] += (gradient[spikes].mean().item() if spikes.any() else 0.0) * inputs.numel() / batches
            quiet = (count.detach() == 0)
            worth[key][name][1] += gradient[quiet].mean().item() * inputs.numel() / batches
    for name, count in zip(names, kept, strict=True):
        firing[name] += (count.detach() > 0).float().mean().item() / batches
print(f"{sys.argv[1]}: language {language.item() / 0.6931:.4f} bpb, {pulses.item():.0f} pulses per byte, budget {checkpoint['args'].get('pulse_budget')}")
for name in names:
    row = " | ".join(f"{key} {worth[key][name][0]:+.1e}/{worth[key][name][1]:+.1e}" for key in worth)
    print(f"  after block {name}: firing {firing[name]:.3f} | worth of one more spike: {row}")
budget = checkpoint["args"].get("pulse_budget") or 310000.0
for name in names:
    task, pulse = worth["task"][name][0], worth["pulses"][name][0]
    print(f"  after block {name}: a firing cell's spike changes the task loss by {task:+.2e} nats and the priced pulses by "
          f"{pulse:+.2f}; the price outweighs the task above a spike price of {abs(task) * budget / max(abs(pulse), 1e-9):.2f}")


@torch.no_grad()
def firing_after(steps):
    saved = [weight.detach().clone() for weight in receives]
    for weight, step in zip(receives, steps, strict=True):
        weight.add_(step)
    probe = torch.Generator().manual_seed(23)
    totals = {name: 0.0 for name in names}
    for _ in range(2):
        inputs, _ = random_batch(train, 8, 512, probe)
        pulse_counts.clear()
        model(inputs.to(device))
        for name in names:
            totals[name] += (cells[name] > 0).float().mean().item() / 2
    for weight, value in zip(receives, saved, strict=True):
        weight.copy_(value)
    return " ".join(f"{totals[name]:.3f}" for name in names)


zero = [torch.zeros_like(weight) for weight in receives]
print(f"  firing now                               : {firing_after(zero)}")
reach = 3e-4 * 50
for key in first:
    steps = [-reach * mean / (square.sqrt() + 1e-12) for mean, square in zip(first[key], second[key], strict=True)]
    agreement = sum((mean.abs() / (square.sqrt() + 1e-12)).mean().item() for mean, square in
                    zip(first[key], second[key], strict=True)) / len(names)
    print(f"  after 50 steps along the {key:8s} gradient : {firing_after(steps)}   (agreement of the batches {agreement:.2f})")
for price in (0.1, 1.0, 10.0):
    for extra, label in ((0.0, "task + price on downstream pulses"), (1.0, "task + price on those and the cells")):
        steps = []
        for index in range(len(names)):
            mean = (first["task"][index] + price / budget * (first["pulses"][index] + extra * first["own"][index]))
            # The second moment of a sum is not the sum of second moments; the task part sets the scale here.
            scale = (second["task"][index] + (price / budget) ** 2 * (second["pulses"][index] + extra * second["own"][index])).sqrt()
            steps.append(-reach * mean / (scale + 1e-12))
        print(f"  50 steps, spike price {price:5.1f}, {label}: {firing_after(steps)}")
