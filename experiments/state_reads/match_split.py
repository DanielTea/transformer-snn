"""Where does one checkpoint beat another? Bits per target split by the longest earlier match of the bytes just read.

usage: match_split.py CHECKPOINT... [--windows 256]     (run from the repository root; the first checkpoint is the reference)
A checkpoint written as PATH@cut is scored with the output synapses of its granule cells cut.
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
from snnformer.event_lm import EventLM  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("checkpoints", nargs="+")
parser.add_argument("--windows", type=int, default=256)
parser.add_argument("--averaged", action="store_true", help="score the running weight average where a checkpoint stores one")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, validation, _ = load_splits(Path("data/enwik8"))
data = np.array(validation)
starts = np.linspace(0, len(data) - 514, args.windows).astype(np.int64)
tokens = np.stack([data[start:start + 513] for start in starts]).astype(np.int64)
length, copyable = earlier_matches(tokens, 512)
windows = torch.from_numpy(tokens).to(device)
bits = []
for path in args.checkpoints:
    cut = path.endswith("@cut")
    model, checkpoint = load_checkpoint(Path(path.removesuffix("@cut")), device)
    averaged = checkpoint.get("averaged_parameters") if args.averaged else None
    if averaged is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaged[name].to(parameter))
    if cut:
        with torch.no_grad():
            for layer in model.granule_layers.values():
                layer.send.zero_()
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
    print(f"[{len(bits) - 1}] {path} | steps {checkpoint['args']['steps']} | {'averaged' if averaged is not None else 'raw'} | {bits[-1].mean():.4f} bpb", flush=True)
    del model
position = np.arange(512)[None].repeat(len(tokens), 0)
total = bits[0].size
print(f"\n{'targets':<34} {'share':>6} " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(len(bits)))
      + " | bpb over [0] times share: " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(1, len(bits))))


def row(name, mask):
    share = mask.sum() / total
    means = [b[mask].mean() for b in bits]
    gaps = [(b[mask].sum() - bits[0][mask].sum()) / total for b in bits[1:]]
    print(f"{name:<34} {share:6.3f} " + " ".join(f"{m:8.4f}" for m in means) + " |" + " " * 27
          + " ".join(f"{g:+8.4f}" for g in gaps))


row("all", np.ones_like(copyable))
for low, high in ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 9999)):
    inside = (length >= low) & (length <= high)
    label = f"match {low}" + ("" if high == low else "+" if high > 999 else f"-{high}")
    if low == 0:
        row(label, inside)
        continue
    row(label + ", follower is the target", inside & copyable)
    row(label + ", follower is not", inside & ~copyable)
row("copyable (match >= 4)", (length >= 4) & copyable)
row("not copyable or match < 4", ~((length >= 4) & copyable))
for low, high in ((0, 31), (32, 127), (128, 255), (256, 511)):
    row(f"bytes before the target {low}-{high}", (position >= low) & (position <= high))
