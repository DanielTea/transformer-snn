"""How consistent the task and the pulse gradients are from batch to batch, per group of weights.

usage: probe_coherence.py CHECKPOINT [BATCHES]   (run from the repository root)
The agreement is |mean gradient| / root mean square gradient over the batches, averaged over a group's weights.
Pure noise gives about 0.8 / sqrt(batches).
"""
import re
import sys
from collections import defaultdict
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
pulse_counts = []
for module in model.modules():
    if isinstance(module, _IntegrateFire):
        module.register_forward_hook(lambda module, _inputs, counts: pulse_counts.append(emitted_pulses(module, counts)))
named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
parameters = [parameter for _, parameter in named]
first = {key: [torch.zeros_like(p) for p in parameters] for key in ("task", "pulses")}
second = {key: [torch.zeros_like(p) for p in parameters] for key in first}
generator = torch.Generator().manual_seed(11)
for _ in range(batches):
    inputs, labels = random_batch(train, 8, 512, generator)
    inputs, labels = inputs.to(device), labels.to(device)
    pulse_counts.clear()
    counts, predicted, observed, *early = model(inputs, return_prediction=True, memory_pulses=pulse_counts,
                                                early_readout=cfg.clock_start)
    keys = cfg.forecast_key_channels if cfg.learn_context_keys else 0
    code = F.smooth_l1_loss(predicted[:, :-1, keys:] / cfg.resolution, observed[:, 1:, keys:].detach() / cfg.resolution)
    language = F.cross_entropy(scoring_logits(counts, True, cfg.resolution).flatten(0, 1), labels.flatten())
    early_loss = F.cross_entropy(scoring_logits(early[0], True, model.forecast_resolution).flatten(0, 1), labels.flatten())
    parts = {"task": language + 0.25 * code + 0.25 * early_loss, "pulses": torch.stack(pulse_counts).sum() / inputs.numel()}
    for key, part in parts.items():
        for index, gradient in enumerate(torch.autograd.grad(part, parameters, retain_graph=True, allow_unused=True)):
            if gradient is not None:
                first[key][index] += gradient / batches
                second[key][index] += gradient.square() / batches
groups = defaultdict(lambda: {"task": [0.0, 0.0, 0.0], "pulses": [0.0, 0.0, 0.0], "size": 0})
for index, (name, parameter) in enumerate(named):
    group = re.sub(r"^blocks\.(\d+)\.", lambda m: "blocks.early." if int(m.group(1)) < 5 else "blocks.late.", name)
    group = re.sub(r"^(granule_layers|state_reads|reads)\.\d+\.", r"\1.*.", group)
    groups[group]["size"] += parameter.numel()
    for key in first:
        mean, square = first[key][index], second[key][index]
        live = square > 0
        if live.any():
            row = groups[group][key]
            row[0] += (mean[live].abs() / square[live].sqrt()).sum().item()
            row[1] += live.sum().item()
            row[2] += square.sum().item()
print(f"{sys.argv[1]}: {batches} batches, noise floor about {0.8 / batches ** 0.5:.2f}")
budget = checkpoint["args"].get("pulse_budget") or 310000.0
for group, row in sorted(groups.items(), key=lambda item: -item[1]["size"])[:26]:
    task, pulses = row["task"], row["pulses"]
    ratio = (pulses[2] / max(task[2], 1e-30)) ** 0.5 / budget
    print(f"  {group:44s} {row['size']:9d} weights | agreement task {task[0] / max(task[1], 1):.2f} pulses "
          f"{pulses[0] / max(pulses[1], 1):.2f} | pulse gradient at spike price 1 is {ratio:.2f} of the task gradient")
