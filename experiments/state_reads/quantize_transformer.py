"""What a trained Transformer loses when its weights are rounded to a coarse grid, as the decoder's synapses are.

usage: quantize_transformer.py CHECKPOINT   (run from the repository root)
Every weight matrix is rounded to multiples of its mean magnitude divided by LEVELS, without retraining.
The decoder's synapses have a mean magnitude of about 3 to 10 ticks.
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
model, _ = load_checkpoint(sys.argv[1], device)
model.eval()


def score():
    return evaluate(model, validation, context=512, batches=16, batch_size=8, device=device, event=False)["bpb"]


saved = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
matrices = [name for name, parameter in model.named_parameters() if parameter.ndim >= 2]
weights = sum(saved[name].numel() for name in matrices)
whole = score()
print(f"{sys.argv[1]}: {whole:.4f} bpb, {weights} weights in {len(matrices)} matrices of {sum(v.numel() for v in saved.values())}")
for levels in (40.0, 20.0, 10.0, 7.0, 5.0, 3.0, 2.0, 1.0):
    with torch.no_grad():
        distinct = []
        for name, parameter in model.named_parameters():
            if name in matrices:
                tick = saved[name].abs().mean() / levels
                parameter.copy_((saved[name] / tick).round() * tick)
                distinct.append((parameter == 0).float().mean().item())
        rounded = score()
        for name, parameter in model.named_parameters():
            parameter.copy_(saved[name])
    print(f"  mean magnitude of {levels:4.0f} ticks: {rounded - whole:+.4f} bpb, {sum(distinct) / len(distinct):.2f} of the weights round to zero")
