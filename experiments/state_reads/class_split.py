"""Where does one checkpoint beat another? Bits per target split by the kind of byte and by where it stands.

usage: class_split.py CHECKPOINT... [--windows 256] [--averaged]   (run from the repository root; the first is the reference)
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
parser.add_argument("--windows", type=int, default=256)
parser.add_argument("--averaged", action="store_true")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, args.windows).astype(np.int64)
tokens = np.stack([data[start:start + 513] for start in starts]).astype(np.int64)
windows = torch.from_numpy(tokens).to(device)

kinds = np.full(256, 8)
for index, group in enumerate((b"abcdefghijklmnopqrstuvwxyz", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ", b"0123456789", b" ", b"\n",
                               b".,;:!?'\"()-", b"[]{}|<>=&/*#_")):
    kinds[list(group)] = index
kinds[128:] = 7
kind_names = ("lower-case letter", "capital letter", "digit", "space", "newline", "punctuation", "markup sign",
              "byte of a longer character", "other")
letter = kinds < 2
target = tokens[:, 1:]
kind = kinds[target]
# Letters that stand right before the target, and the bracket the target stands inside.
run = np.zeros_like(target)
region = np.zeros_like(target)
for row in range(tokens.shape[0]):
    count, tag, link, template = 0, False, 0, 0
    for column in range(512):
        byte, earlier = tokens[row, column], tokens[row, column - 1] if column else -1
        count = count + 1 if letter[byte] else 0
        if byte == 60:
            tag = True
        elif byte == 62:
            tag = False
        elif byte == 91 and earlier == 91:
            link += 1
        elif byte == 93 and earlier == 93:
            link = max(link - 1, 0)
        elif byte == 123 and earlier == 123:
            template += 1
        elif byte == 125 and earlier == 125:
            template = max(template - 1, 0)
        run[row, column] = count
        region[row, column] = 1 if tag else 2 if link else 3 if template else 0

bits = []
for path in args.checkpoints:
    model, checkpoint = load_checkpoint(Path(path), device)
    averaged = checkpoint.get("averaged_parameters") if args.averaged else None
    if averaged is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaged[name].to(parameter))
    model.eval()
    event = isinstance(model, EventLM)
    resolution = model.cfg.resolution if event else 1
    rows = []
    with torch.inference_mode():
        for batch in windows.split(16):
            logits = scoring_logits(model(batch[:, :-1]), event, resolution)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="none")
            rows.append(loss.view(batch.shape[0], -1).cpu().numpy() / math.log(2))
    bits.append(np.concatenate(rows))
    print(f"[{len(bits) - 1}] {path} | steps {checkpoint['args']['steps']} | {'averaged' if averaged is not None else 'raw'} "
          f"| {bits[-1].mean():.4f} bpb", flush=True)
    del model
total = bits[0].size
print(f"\n{'targets':<44} {'share':>6} " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(len(bits)))
      + " | bpb over [0] times share: " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(1, len(bits)))
      + " | bpb over [0]: " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(1, len(bits))))


def row(name, mask):
    if not mask.any():
        return
    share = mask.sum() / total
    means = [b[mask].mean() for b in bits]
    gaps = [(b[mask].sum() - bits[0][mask].sum()) / total for b in bits[1:]]
    print(f"{name:<44} {share:6.3f} " + " ".join(f"{m:8.4f}" for m in means) + " |" + " " * 27
          + " ".join(f"{g:+8.4f}" for g in gaps) + " |" + " " * 15 + " ".join(f"{m - means[0]:+8.4f}" for m in means[1:]))


row("all", np.ones_like(target, dtype=bool))
for index, name in enumerate(kind_names):
    row(name, kind == index)
is_letter = kind < 2
for low, high in ((0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 999)):
    label = f"letter after {low}" + ("" if high == low else "+" if high > 99 else f"-{high}") + " letters"
    row(label, is_letter & (run >= low) & (run <= high))
row("not a letter, after a letter (a word ends)", ~is_letter & (run > 0))
row("not a letter, after another", ~is_letter & (run == 0))
for index, name in enumerate(("plain text", "inside a tag <...>", "inside a link [[...]]", "inside a template {{...}}")):
    row(name, region == index)
    row("  letters there", (region == index) & is_letter)
position = np.arange(512)[None].repeat(len(tokens), 0)
for low, high in ((0, 31), (32, 127), (128, 511)):
    here = (position >= low) & (position <= high)
    row(f"bytes before {low}-{high}: first letter of a word", here & is_letter & (run == 0))
    row(f"bytes before {low}-{high}: letter after 1-2", here & is_letter & (run >= 1) & (run <= 2))
    row(f"bytes before {low}-{high}: letter after 3+", here & is_letter & (run >= 3))
    row(f"bytes before {low}-{high}: not a letter", here & ~is_letter)
