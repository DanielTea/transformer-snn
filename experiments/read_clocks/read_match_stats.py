"""How fine are the probe and tag pulses of a trained decoder's reads, and how far does the match spread the cells?

usage: read_match_stats.py CHECKPOINT [--windows 16]      (run from the repository root)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _PACKET_UNITS, _RACE_UNIT  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--windows", type=int, default=16)
args = parser.parse_args()
torch.set_num_threads(4)
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, args.windows).astype(np.int64)
tokens = torch.from_numpy(np.stack([data[start:start + 512] for start in starts]).astype(np.int64))
model, _ = load_checkpoint(Path(args.checkpoint), torch.device("cpu"))
model.eval()
seen = {}
for index, read in model.state_reads.items():
    read.register_forward_pre_hook(lambda module, inputs, index=index: seen.__setitem__(index, inputs[0]))
with torch.inference_mode():
    model(tokens)
    steps = tokens.shape[1]
    distance = torch.arange(steps)[:, None] - torch.arange(steps)
    visible = (distance >= 0) & (distance < 512)
    for index, read in model.state_reads.items():
        spikes = seen[index]
        quanta = (_PACKET_UNITS + read.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))) * read.threshold_scale
        probe, tag = read.probe(spikes), read.tag(spikes)
        if not read.signed:
            probe, tag = probe.clamp_min(0), tag.clamp_min(0)
        probe, tag = read._split(read.fire(probe, quanta)), read._split(read.fire(tag, quanta))
        payload = read._split(read.fire(read.payload(spikes), quanta))
        unit = read.resolution * math.sqrt(probe.shape[-1]) * (_RACE_UNIT if read.signed else 0.5)
        match = probe @ tag.transpose(2, 3) * ((read.gain * read.rate).exp() / unit)
        ages = torch.nn.functional.pad(read.age[:, :steps] * read.rate, (0, max(0, steps - read.window))).flip(-1)
        offset = torch.nn.functional.pad(ages[:, None].expand(-1, steps, -1), (1, 0)).reshape(read.heads, steps + 1, steps)[:, 1:]
        print(f"read after block {index}: stream |count| mean {spikes.abs().mean():.2f}, zero share {(spikes == 0).float().mean():.2f}")
        for name, packet in (("probe", probe), ("tag", tag), ("payload", payload)):
            per_head = packet.abs().mean((0, 2, 3))
            zero = (packet == 0).float().mean((0, 2, 3))
            large = (packet.abs() >= 4).float().mean((0, 2, 3))
            print(f"  {name:<8} pulses per channel " + " ".join(f"{v:.2f}" for v in per_head.tolist())
                  + " | silent share " + " ".join(f"{v:.2f}" for v in zero.tolist())
                  + " | four or more " + " ".join(f"{v:.2f}" for v in large.tolist()))
        # At the last quarter of each window, how far apart are the cells' charges?
        rows = slice(384, 512)
        mask = visible[rows]
        for name, values in (("match", match[:, :, rows]), ("age offset", offset[None, :, rows].expand(match.shape[0], -1, -1, -1)),
                             ("both", match[:, :, rows] + offset[None, :, rows])):
            inside = values.masked_fill(~mask, float("nan"))
            spread = torch.nan_to_num(inside, nan=0.0)
            count = mask.sum(-1)
            mean = spread.sum(-1) / count
            deviation = ((spread - mean[..., None]).square() * mask).sum(-1).div(count).sqrt().mean((0, 2))
            top = (inside.nan_to_num(-1e9).amax(-1) - mean).mean((0, 2))
            entropy = torch.special.entr(values.masked_fill(~mask, -torch.inf).softmax(-1)).sum(-1)
            cells = entropy.exp().mean((0, 2))
            print(f"  {name:<10} spread over cells " + " ".join(f"{v:.2f}" for v in deviation.tolist())
                  + " | leader above mean " + " ".join(f"{v:.1f}" for v in top.tolist())
                  + " | cells that share the read " + " ".join(f"{v:.0f}" for v in cells.tolist()))
