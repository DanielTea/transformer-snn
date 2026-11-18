"""Where does one checkpoint fall behind another as the context grows? Bits per target by the bytes in view.

usage: context_split.py CHECKPOINT... [--long 512] [--short 4096] [--threads 24] [--save FILE]
Run from the repository root; the first checkpoint is the reference. A long window gives targets with 1 to 512 bytes in
view, a short one targets with 1 to 64, so the short contexts are sampled as densely as the long ones.
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
from snnformer.event_lm import EventLM  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--long", type=int, default=512)
parser.add_argument("--short", type=int, default=4096)
parser.add_argument("--threads", type=int, default=24)
parser.add_argument("--save")
args = parser.parse_args()
torch.set_num_threads(args.threads)
device = torch.device("cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)


def cut(count, length, shift):
    starts = np.linspace(0, len(data) - length - 2 - shift, count).astype(np.int64) + shift
    return np.stack([data[start:start + length + 1] for start in starts]).astype(np.int64)


sets = [cut(args.long, 512, 0), cut(args.short, 64, 257)]
kinds = np.full(256, 8)
for index, group in enumerate((b"abcdefghijklmnopqrstuvwxyz", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"0123456789", b" ", b"\n",
                               b".,;:!?'\"()-", b"[]{}|<>=&/*#_")):
    kinds[list(group)] = index
kinds[128:] = 7
letter = kinds < 2


def features(tokens):
    """For every target: bytes in view, letters right before it, words that ended in view, and whether it is a letter."""
    rows, length = tokens.shape[0], tokens.shape[1] - 1
    view = np.arange(1, length + 1)[None].repeat(rows, 0)
    run = np.zeros((rows, length), dtype=np.int64)
    words = np.zeros((rows, length), dtype=np.int64)
    for row in range(rows):
        count = ended = 0
        for column in range(length):
            if letter[tokens[row, column]]:
                count += 1
            else:
                ended += count > 0
                count = 0
            run[row, column] = count
            words[row, column] = ended
    return view, run, words, letter[tokens[:, 1:]], kinds[tokens[:, 1:]]


parts = [features(tokens) for tokens in sets]
view, run, words, is_letter, kind = (np.concatenate([part[index].ravel() for part in parts]) for index in range(5))
source = np.concatenate([np.full(part[0].size, index) for index, part in enumerate(parts)])

bits = []
for path in args.checkpoints:
    model, checkpoint = load_checkpoint(Path(path), device)
    model.eval()
    event = isinstance(model, EventLM)
    resolution = model.cfg.resolution if event else 1
    rows = []
    with torch.inference_mode():
        for tokens, size in zip(sets, (16, 128)):
            for batch in torch.from_numpy(tokens).split(size):
                logits = scoring_logits(model(batch[:, :-1]), event, resolution)
                loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="none")
                rows.append(loss.numpy().ravel() / math.log(2))
    bits.append(np.concatenate(rows))
    print(f"[{len(bits) - 1}] {path} | steps {checkpoint['args']['steps']} | long windows "
          f"{bits[-1][source == 0].mean():.4f} bpb | short windows {bits[-1][source == 1].mean():.4f} bpb", flush=True)
    del model
if args.save:
    np.savez_compressed(args.save, bits=np.stack(bits), view=view, run=run, words=words, kind=kind, source=source)


def row(name, mask):
    count = int(mask.sum())
    if count < 50:
        return
    means = [b[mask].mean() for b in bits]
    out = f"{name:<46} {count:>8} " + " ".join(f"{m:7.4f}" for m in means) + " |"
    for b in bits[1:]:
        difference = b[mask] - bits[0][mask]
        out += f" {difference.mean():+7.4f} ±{difference.std() / math.sqrt(count):6.4f}"
    print(out)


def table(title, mask, bins, values, label):
    print(f"\n{title}\n{'':<46} {'targets':>8} " + " ".join(f"{'[' + str(i) + ']':>7}" for i in range(len(bits)))
          + " | over [0], with its standard error")
    for low, high in bins:
        name = f"{label} {low}" + ("" if high == low else "+" if high > 9000 else f"-{high}")
        row(name, mask & (values >= low) & (values <= high))


everything = np.ones_like(is_letter)
fine = [(k, k) for k in range(1, 9)] + [(9, 12), (13, 16), (17, 24), (25, 32), (33, 48), (49, 64), (65, 128),
                                         (129, 256), (257, 512)]
coarse = [(1, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512)]
word_bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 19), (20, 39), (40, 9999)]
first = is_letter & (run == 0)
table("All targets by the bytes in view", everything, fine, view, "bytes in view")
table("Letters by the bytes in view", is_letter, fine, view, "bytes in view")
table("Other bytes by the bytes in view", ~is_letter, coarse, view, "bytes in view")
table("First letter of a word by the bytes in view", first, coarse, view, "bytes in view")
table("First letter of a word by the words that ended in view", first, word_bins, words, "earlier words")
table("Letter after 1-2 letters by the words that ended in view", is_letter & (run >= 1) & (run <= 2), word_bins, words,
      "earlier words")
table("Letter after 3+ letters by the words that ended in view", is_letter & (run >= 3), word_bins, words,
      "earlier words")
table("Whole view is the present word: letters by the bytes in view", is_letter & (run == view),
      [(k, k) for k in range(1, 9)] + [(9, 16)], view, "bytes in view")
table("Letter after 3+ letters by the letters before it", is_letter, [(3, 3), (4, 4), (5, 5), (6, 7), (8, 11), (12, 9999)],
      run, "letters before")
long_only = source == 0
print("\nShare of the long windows' gap that each stretch of context lengths carries")
for index in range(1, len(bits)):
    difference = (bits[index] - bits[0])[long_only]
    total = difference.sum()
    out = f"[{index}] mean gap {difference.mean():+.4f}:"
    for low, high in coarse:
        inside = (view[long_only] >= low) & (view[long_only] <= high)
        out += f" {low}-{high}: {difference[inside].mean():+.4f}"
    print(out)
