"""What does the rounding to whole spikes cost a trained decoder?

Every firing population rounds its charge to whole spikes at a fixed phase. Here the phase of each neuron and byte is
drawn at random instead (a dither of up to half a threshold either way), which doubles the variance of the rounding
error without a bias. The loss of one such pass over the decoder's own loss is what one more helping of rounding noise
costs, and the loss of the averaged predictions of several passes shows how much of the loss is noise that averages out.

usage: probe_dither.py [--averaged] [--windows 128] [--passes 8] [--only PART,...] CHECKPOINT...   (run from the repository root)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _IntegrateFire, _RaceFire  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--averaged", action="store_true")
parser.add_argument("--windows", type=int, default=128)
parser.add_argument("--passes", type=int, default=8)
parser.add_argument("--scale", type=float, default=1.0, help="width of the dither in thresholds")
parser.add_argument("--only", default="", help="comma-separated name parts; each is dithered alone as well")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, args.windows).astype(np.int64)
tokens = torch.from_numpy(np.stack([data[start:start + 513] for start in starts]).astype(np.int64)).to(device)

active = {"names": None, "generator": None}
plain = _IntegrateFire.forward


def dithered(self, current, quanta=None, *, reference=None):
    names = active["names"]
    if names is None or isinstance(self, _RaceFire) or not current.is_floating_point() or (
            names and not any(part in self.probe_name for part in names)):
        return plain(self, current, quanta, reference=reference)
    step = self.threshold if quanta is None else quanta * (self.threshold // self.quanta_per_threshold)
    noise = torch.rand(current.shape, device=current.device, generator=active["generator"]) - 0.5
    return plain(self, current + (noise * args.scale * step).floor(), quanta, reference=reference)


_IntegrateFire.forward = dithered


def log_probabilities(model, names, seed):
    active["names"] = names
    active["generator"] = None if names is None else torch.Generator(device=device).manual_seed(seed)
    rows = []
    with torch.inference_mode():
        for start in range(0, tokens.shape[0], 32):
            batch = tokens[start:start + 32]
            logits = scoring_logits(model(batch[:, :-1]), True, model.cfg.resolution)
            rows.append(F.log_softmax(logits.float(), -1))
    active["names"] = None
    return torch.cat(rows)


def bits(log_probability):
    return -(log_probability.gather(-1, tokens[:, 1:, None]).mean().item()) / math.log(2)


for path in args.checkpoints:
    model, checkpoint = load_checkpoint(path, device)
    averaged = checkpoint.get("averaged_parameters") if args.averaged else None
    if averaged is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaged[name].to(parameter))
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, _IntegrateFire):
            module.probe_name = name
    own = log_probabilities(model, None, 0)
    again = log_probabilities(model, None, 0)
    assert torch.equal(own, again)
    base = bits(own)
    print(f"{path} | {'averaged' if averaged is not None else 'raw'} | own loss {base:.4f} bpb", flush=True)
    for label, names in [("all populations", ())] + [(part, (part,)) for part in args.only.split(",") if part]:
        passes = [log_probabilities(model, names, seed) for seed in range(1, args.passes + 1)]
        single = float(np.mean([bits(one) for one in passes]))
        mixture = torch.logsumexp(torch.stack(passes), 0) - math.log(len(passes))
        both = torch.logsumexp(torch.stack(passes + [own]), 0) - math.log(len(passes) + 1)
        change = float(np.mean([(own.exp() * (own - one)).sum(-1).mean().item() for one in passes])) / math.log(2)
        print(f"  dither on {label:<22} one pass {single:.4f} ({single - base:+.4f}) | {len(passes)} passes averaged "
              f"{bits(mixture):.4f} ({bits(mixture) - base:+.4f}) | with the own pass {bits(both):.4f} | "
              f"KL(own, dithered) {change:.4f} bits", flush=True)
    del model
