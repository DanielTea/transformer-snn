"""Do the cells use their delayed fibres? usage: tap_use.py [--averaged] CHECKPOINT...   (run from the repository root)

Per granule layer: the mean size of the cells' input synapses on the fibres of each delay, and the validation
loss when the fibres of one delay are silenced.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _weight_ticks  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

use_average = "--averaged" in sys.argv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, 128).astype(np.int64)
windows = torch.from_numpy(np.stack([data[start:start + 513] for start in starts]).astype(np.int64)).to(device)
for path in [argument for argument in sys.argv[1:] if not argument.startswith("--")]:
    model, checkpoint = load_checkpoint(Path(path), device)
    averaged = checkpoint.get("averaged_parameters") if use_average else None
    if averaged is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaged[name].to(parameter))
    model.eval()

    @torch.inference_mode()
    def loss():
        total = 0.0
        for batch in windows.split(16):
            logits = scoring_logits(model(batch[:, :-1]), True, model.cfg.resolution)  # noqa: F821
            total += F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="sum").item()
        return total / (windows.shape[0] * 512) / math.log(2)

    base = loss()
    delays = model.cfg.granule_delays
    print(f"{path} | steps {checkpoint['args']['steps']} | {'averaged' if averaged is not None else 'raw'} | {base:.4f} bpb")
    for index, layer in model.granule_layers.items():
        ticks = _weight_ticks(layer.receive, charge_bits=layer.bits).abs()
        sizes = [ticks[(layer.taps == delay)[:, None].expand_as(ticks)].mean().item() for delay in range(delays)]
        print(f"  layer after block {index}: mean |ticks| by delay " + " ".join(f"{size:6.2f}" for size in sizes))
    saved = {index: layer.receive.detach().clone() for index, layer in model.granule_layers.items()}
    for delay in range(delays):
        with torch.no_grad():
            for layer in model.granule_layers.values():
                layer.receive.mul_((layer.taps != delay)[:, None].to(layer.receive.dtype))
        print(f"  without the fibres of delay {delay}: {loss() - base:+.4f} bpb", flush=True)
        with torch.no_grad():
            for index, layer in model.granule_layers.items():
                layer.receive.copy_(saved[index])
    del model
