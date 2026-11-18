"""How much of each synapse's input is a constant per channel? usage: common_mode.py CHECKPOINT (run from the repository root)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _Synapse  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, checkpoint = load_checkpoint(Path(sys.argv[1]), device)
model.eval()
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, 32).astype(np.int64)
windows = torch.from_numpy(np.stack([data[start:start + 512] for start in starts]).astype(np.int64)).to(device)
stats = {}


def record(name):
    def hook(module, inputs):
        x = inputs[0].detach().float()
        if x.dim() < 2 or x.shape[-1] != module.weight.shape[1]:
            return
        x = x.reshape(-1, x.shape[-1])
        entry = stats.setdefault(name, [0, torch.zeros(x.shape[-1], device=device), torch.zeros(x.shape[-1], device=device)])
        entry[0] += x.shape[0]
        entry[1] += x.sum(0)
        entry[2] += x.square().sum(0)
    return hook


for name, module in model.named_modules():
    if isinstance(module, _Synapse):
        module.register_forward_pre_hook(record(name))
with torch.inference_mode():
    for batch in windows.split(16):
        model(batch)
print(f"{'synapse':<34} {'inputs':>7} {'mean|x|':>8} {'rms':>8} {'constant share of the energy':>30} {'channels mostly one sign':>26}")
for name, (count, total, squares) in stats.items():
    mean, second = total / count, squares / count
    share = (mean.square().sum() / second.sum().clamp_min(1e-9)).item()
    variance = (second - mean.square()).clamp_min(0)
    one_sign = ((mean.abs() > variance.sqrt()).float().mean()).item()
    print(f"{name:<34} {mean.numel():>7} {second.sqrt().mean().item():8.2f} {second.mean().sqrt().item():8.2f} "
          f"{share:30.3f} {one_sign:26.3f}")
