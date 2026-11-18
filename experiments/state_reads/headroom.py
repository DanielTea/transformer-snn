"""How much of the exact-integer accumulator range saved decoders use. usage: headroom.py CHECKPOINT..."""
import sys

import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer import event_lm  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402

for path in sys.argv[1:]:
    model, _ = load_checkpoint(path, torch.device("cpu"))
    limit = event_lm._FP32_EXACT // model.input_emit.cap
    rows = {}
    for index, block in enumerate(model.blocks):
        expand = block.expand.weight_ticks().abs().sum(-1).amax().item()
        units = event_lm._PACKET_UNITS if block.packet_thresholds is not None else 1
        project = block.project.weight_ticks().abs().sum(-1) * units + block.emit.threshold
        if block.arrival_project is not None:
            project = project + block.arrival_project.weight_ticks().abs().sum(-1) * units
        rows[index] = (expand / limit, project.amax().item() / limit,
                       block.expand.weight_ticks().abs().mean().item(), block.project.weight_ticks().abs().mean().item())
    worst = max(max(row[:2]) for row in rows.values())
    print("packet units:", [event_lm._PACKET_UNITS if b.packet_thresholds is not None else 1 for b in model.blocks][:3])
    print(f"{path}: bits {model.cfg.router_charge_bits} | limit {limit} | worst share of the limit {worst:.3f}")
    print("  expand share by block: " + " ".join(f"{row[0]:.2f}" for row in rows.values()))
    print("  project share by block: " + " ".join(f"{row[1]:.2f}" for row in rows.values()))
    print("  mean |ticks| expand: " + " ".join(f"{row[2]:.1f}" for row in rows.values()))
    print("  mean |ticks| project: " + " ".join(f"{row[3]:.1f}" for row in rows.values()))
    try:
        model._check_accumulator()
        print("  the guard passes")
    except ValueError as error:
        print("  the guard fails:", error)
