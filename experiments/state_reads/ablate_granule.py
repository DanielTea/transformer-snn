"""What the granule layers of a saved decoder are worth: validation bpb with their output synapses cut.

usage: ablate_granule.py [--averaged] CHECKPOINT...   (run from the repository root)
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from train_event_lm import evaluate, load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
paths = [argument for argument in sys.argv[1:] if argument != "--averaged"]
for path in paths:
    model, checkpoint = load_checkpoint(path, device)
    label = "raw"
    if "--averaged" in sys.argv:
        averaged = torch.load(path, map_location=device, weights_only=False).get("averaged_parameters")
        if averaged is not None:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(averaged[name])
            label = "averaged"
    model.eval()

    def score():
        return evaluate(model, validation, context=512, batches=16, batch_size=8, device=device, event=True)["bpb"]

    whole = score()
    print(f"{path} ({label}): {whole:.4f} bpb")
    saved = {name: layer.send.detach().clone() for name, layer in model.granule_layers.items()}
    with torch.no_grad():
        for name, layer in model.granule_layers.items():
            layer.send.zero_()
            print(f"  without the cells after block {name}: {score() - whole:+.4f}")
            layer.send.copy_(saved[name])
        for layer in model.granule_layers.values():
            layer.send.zero_()
        print(f"  without any cells: {score() - whole:+.4f}")
