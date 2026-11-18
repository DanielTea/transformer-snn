"""Caps and stream activity of saved decoders. usage: stream_stats.py [--averaged] CHECKPOINT...   (run from the repository root)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _IntegrateFire  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

args = sys.argv[1:]
averaged = "--averaged" in args
paths = [arg for arg in args if arg != "--averaged"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, 16).astype(np.int64)
tokens = torch.from_numpy(np.stack([data[start:start + 512] for start in starts]).astype(np.int64)).to(device)
for path in paths:
    model, checkpoint = load_checkpoint(path, device)
    if averaged:
        mean = checkpoint.get("averaged_parameters")
        if mean is None:
            print(path, "has no averaged weights")
            continue
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(mean[name].to(parameter))
    model.eval()
    fire, streams = {}, {}

    def watch_fire(name):
        def hook(layer, inputs, output):
            current = inputs[0].float()
            counts = output.float()
            edge = layer.threshold * layer.cap
            row = fire.setdefault(name, [0.0, 0.0, 0.0, 0.0, 0])
            row[0] += (current.abs() >= edge).float().mean().item()
            row[1] += (counts == 0).float().mean().item()
            row[2] += counts.abs().mean().item()
            row[3] = max(row[3], counts.abs().amax().item())
            row[4] += 1
        return hook

    def watch_block(name):
        def hook(layer, inputs, output):
            spikes = inputs[0].float()
            total = spikes.abs().sum(-1)
            square = spikes.square().sum(-1).sqrt()
            streams[name] = (total.mean().item(), (total.std() / total.mean()).item(),
                             (square.std() / square.mean()).item(), (spikes == 0).float().mean().item(),
                             spikes.abs().amax().item(), total.amin().item(), total.amax().item())
        return hook

    handles = [module.register_forward_hook(watch_fire(name)) for name, module in model.named_modules()
               if isinstance(module, _IntegrateFire)]
    handles += [block.register_forward_hook(watch_block(index)) for index, block in enumerate(model.blocks)]
    with torch.inference_mode():
        model(tokens)
    for handle in handles:
        handle.remove()
    print(path, "(averaged)" if averaged else "")
    print("  stream as each block hears it: pulses per byte, spread over bytes (L1, L2), silent channels, largest count, range")
    for name, row in streams.items():
        print(f"    block {name}: {row[0]:8.0f}  cv {row[1]:.3f} / {row[2]:.3f}  silent {row[3]:.3f}  most {row[4]:.0f}  "
              f"[{row[5]:.0f}, {row[6]:.0f}]")
    print("  firing populations: share of drives at the cap (no gradient), share silent, mean |count|, largest")
    for name, row in fire.items():
        calls = row[4]
        print(f"    {name:34s} capped {row[0] / calls:.4f}  silent {row[1] / calls:.3f}  mean {row[2] / calls:6.2f}  "
              f"most {row[3]:.0f}  ({calls} calls)")
