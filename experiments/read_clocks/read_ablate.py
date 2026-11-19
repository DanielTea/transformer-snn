"""What are the reads of a trained decoder worth, and what is their content match worth?

usage: read_ablate.py CHECKPOINT [--windows 96]      (run from the repository root)

Scores the decoder as it is, with the match of probe and tag silenced in every read (each cell's share then
follows from its age alone), and with every read silenced, and splits the bits by copyable and other targets and
by the number of words in view.
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
from probe_match_gap import earlier_matches  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--windows", type=int, default=96)
args = parser.parse_args()
torch.set_num_threads(6)
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, args.windows).astype(np.int64)
tokens = np.stack([data[start:start + 513] for start in starts]).astype(np.int64)
length, copyable = earlier_matches(tokens, 512)
windows = torch.from_numpy(tokens)
model, checkpoint = load_checkpoint(Path(args.checkpoint), torch.device("cpu"))
model.eval()
resolution = model.cfg.resolution


def score():
    rows = []
    with torch.inference_mode():
        for batch in windows.split(16):
            logits = scoring_logits(model(batch[:, :-1]), True, resolution)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="none")
            rows.append(loss.view(batch.shape[0], -1).numpy() / math.log(2))
    return np.concatenate(rows)


names, bits = ["as trained"], [score()]
print(f"as trained {bits[0].mean():.4f}", flush=True)
reads = list(model.state_reads.values())
saved = [(read.gain.detach().clone(), read.project.weight.detach().clone()) for read in reads]
with torch.no_grad():
    for read in reads:
        read.gain.fill_(-1e6)
names.append("no match")
bits.append(score())
print(f"no match {bits[-1].mean():.4f}", flush=True)
with torch.no_grad():
    for read, (gain, _) in zip(reads, saved):
        read.gain.copy_(gain)
for index, read in enumerate(reads):
    with torch.no_grad():
        read.gain.fill_(-1e6)
    names.append(f"no match in read {index}")
    bits.append(score())
    print(f"{names[-1]} {bits[-1].mean():.4f}", flush=True)
    with torch.no_grad():
        read.gain.copy_(saved[index][0])
with torch.no_grad():
    for read in reads:
        read.project.weight.zero_()
names.append("no reads")
bits.append(score())
print(f"no reads {bits[-1].mean():.4f}", flush=True)

word = np.zeros(256, dtype=bool)
for low, high in ((48, 57), (65, 90), (97, 122), (128, 255)):
    word[low:high + 1] = True
inside = word[tokens]
start = inside.copy()
start[:, 1:] &= ~inside[:, :-1]
# Words that started before each target, and whether the target is a word's first letter.
seen = start.cumsum(1)[:, :-1]
first = start[:, 1:]
total = bits[0].size
print(f"\n{'targets':<34} {'share':>6} " + " ".join(f"{name[:12]:>12}" for name in names))


def row(name, mask):
    print(f"{name:<34} {mask.sum() / total:6.3f} " + " ".join(f"{b[mask].mean():12.4f}" for b in bits))


row("all", np.ones_like(copyable))
row("copyable (match >= 4)", (length >= 4) & copyable)
row("not copyable or match < 4", ~((length >= 4) & copyable))
row("first letters", first)
row("first letters, not copyable", first & ~((length >= 4) & copyable))
for low, high in ((0, 0), (1, 1), (2, 3), (4, 9), (10, 39), (40, 999)):
    row(f"first letters, {low}-{high} words in view", first & (seen >= low) & (seen <= high))
for low, high in ((0, 0), (1, 1), (2, 3), (4, 9), (10, 39), (40, 999)):
    row(f"not copyable, {low}-{high} words in view", ~((length >= 4) & copyable) & (seen >= low) & (seen <= high))
