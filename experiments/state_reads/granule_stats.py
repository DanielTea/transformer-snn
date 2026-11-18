"""How the granule cells of saved decoders fire. usage: granule_stats.py CHECKPOINT...   (run from the repository root)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _weight_ticks  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, 16).astype(np.int64)
tokens = torch.from_numpy(np.stack([data[start:start + 512] for start in starts]).astype(np.int64)).to(device)
for path in sys.argv[1:]:
    model, checkpoint = load_checkpoint(path, device)
    model.eval()
    rows, races = {}, {}

    def watch(name):
        def hook(layer, inputs, output):
            spikes = inputs[0]
            cells = layer.cells(spikes)
            change = output - spikes
            rows[name] = (
                (cells > 0).float().mean().item(), cells.mean().item(), cells.amax().item(),
                ((cells > 0).float().mean((0, 1)) == 0).float().mean().item(),
                change.abs().mean().item(), spikes.abs().mean().item(),
                _weight_ticks(layer.send, charge_bits=layer.bits).abs().mean().item(),
                _weight_ticks(layer.receive, charge_bits=layer.bits).abs().mean().item(),
                layer.rest.mean().item(), layer.rest.min().item(), layer.rest.max().item(), cells[0].numel() // cells.shape[1],
            )
            if getattr(layer, "budget", None) is not None:
                races[name] = (cells.sum(-1).mean().item(), (cells > 0).float().sum(-1).mean().item(),
                               (cells.amax(-1) / cells.sum(-1).clamp_min(1)).mean().item(),
                               layer.sharpness.exp().min().item(), layer.sharpness.exp().max().item())
        return hook

    handles = [layer.register_forward_hook(watch(name)) for name, layer in model.granule_layers.items()]
    with torch.inference_mode():
        model(tokens)
    for handle in handles:
        handle.remove()
    print(path)
    for name, row in rows.items():
        print(f"  after block {name}: {row[11]} cells | firing {row[0]:.3f} of cell-bytes, {row[1]:.3f} spikes per cell-byte, "
              f"most {row[2]:.0f} | never firing {row[3]:.3f} | stream change {row[4]:.3f} of {row[5]:.2f} pulses per channel | "
              f"|send| {row[6]:.1f} |receive| {row[7]:.1f} ticks | rest {row[8]:.2f} [{row[9]:.2f}, {row[10]:.2f}]")
        if name in races:
            race = races[name]
            print(f"      race: {race[0]:.1f} spikes per cluster-byte over {race[1]:.1f} cells, the leader takes {race[2]:.2f} | "
                  f"sharpness {race[3]:.2f} to {race[4]:.2f}")
