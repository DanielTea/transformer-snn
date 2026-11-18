"""Where does one checkpoint differ from another: bits per target by the words in view, paired on the same windows.

usage: words_in_view.py CHECKPOINT... [--windows 512] [--averaged] [--threads 8] [--json OUT]
Run from the repository root; the first checkpoint is the reference.

Every checkpoint scores the same evenly spaced 512-byte windows of the enwik8 validation split, each from an
empty state. The rows split the targets by whether they can be copied (an earlier match of four or more bytes
whose follower is the target), by their place in a word, and by the number of words that started in the window
before them. The rows by words in view hold only targets that cannot be copied. The intervals resample whole
windows.
"""
import argparse
import json
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

VIEWS = ((0, 0), (1, 1), (2, 3), (4, 9), (10, 39), (40, 9999))


def arguments(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--windows", type=int, default=512)
    parser.add_argument("--averaged", action="store_true",
                        help="score the running weight average where a checkpoint stores one")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    return parser.parse_args()


def validation_windows(count):
    """Evenly spaced windows of 513 bytes of the enwik8 validation split: 512 inputs and their targets."""
    _, validation, _ = load_splits(Path("data/enwik8"))
    data = np.array(validation)
    starts = np.linspace(0, len(data) - 514, count).astype(np.int64)
    return np.stack([data[start:start + 513] for start in starts]).astype(np.int64)


def load_weights(path, averaged, device):
    """A checkpoint's model, with the running weight average in place where asked for and stored."""
    model, checkpoint = load_checkpoint(Path(path), device)
    stored = checkpoint.get("averaged_parameters") if averaged else None
    if stored is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(stored[name].to(parameter))
    return model, stored is not None


def paired_bits(paths, tokens, averaged, device):
    """The bits of every target of the windows under every checkpoint, and the checkpoints' names."""
    windows = torch.from_numpy(tokens).to(device)
    bits, names = [], []
    for path in paths:
        model, took_average = load_weights(path, averaged, device)
        event = isinstance(model, EventLM)
        resolution = model.cfg.resolution if event else 1
        rows = []
        with torch.inference_mode():
            for batch in windows.split(16):
                logits = scoring_logits(model(batch[:, :-1]), event, resolution)
                loss = F.cross_entropy(logits.flatten(0, 1).float(), batch[:, 1:].flatten(), reduction="none")
                rows.append(loss.view(batch.shape[0], -1).cpu().numpy() / math.log(2))
        bits.append(np.concatenate(rows))
        names.append(f"{Path(path).parent.name}/{Path(path).name}")
        print(f"[{len(bits) - 1}] {names[-1]} | {'averaged' if took_average else 'as stored'} | "
              f"{bits[-1].mean():.4f} bpb", flush=True)
        del model
    return bits, names


def word_places(tokens):
    """For every byte: whether it is inside a word, whether it starts one, and its place in its word (0 outside)."""
    word = np.zeros(256, dtype=bool)
    for low, high in ((48, 57), (65, 90), (97, 122), (128, 255)):
        word[low:high + 1] = True
    inside = word[tokens]
    start = inside.copy()
    start[:, 1:] &= ~inside[:, :-1]
    place = np.zeros_like(tokens)
    for column in range(tokens.shape[1]):
        place[:, column] = np.where(inside[:, column], (place[:, column - 1] if column else 0) + 1, 0)
    return inside, start, place


def paired_rows(bits, width=40):
    """A function that prints one row for a mask of targets and keeps it, and the list of the kept rows."""
    generator = np.random.default_rng(0)
    draws = generator.integers(0, len(bits[0]), (2000, len(bits[0])))
    total = bits[0].size
    print(f"\n{'targets':<{width}} {'share':>6} " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(len(bits)))
          + " | over [0], with the 95% interval over windows")
    rows = []

    def row(name, mask):
        count = mask.sum(1)
        if not count.sum():
            return
        means = [float(b[mask].mean()) for b in bits]
        record = {"targets": name.strip(), "share": float(mask.sum() / total), "bits": means, "over_reference": []}
        text = []
        for other in bits[1:]:
            gap = ((other - bits[0]) * mask).sum(1)
            resampled = gap[draws].sum(1) / np.maximum(count[draws].sum(1), 1)
            low, high = np.percentile(resampled, (2.5, 97.5))
            record["over_reference"].append({"mean": float(gap.sum() / count.sum()), "low": float(low),
                                             "high": float(high)})
            text.append(f"{gap.sum() / count.sum():+.4f} ({low:+.4f} to {high:+.4f})")
        rows.append(record)
        print(f"{name:<{width}} {record['share']:6.3f} " + " ".join(f"{m:8.4f}" for m in means)
              + " | " + "  ".join(text))

    return row, rows


def write_rows(path, names, args, rows):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"checkpoints": names, "windows": args.windows, "averaged": args.averaged,
                                    "rows": rows}, indent=1) + "\n")


def main():
    args = arguments(__doc__)
    torch.set_num_threads(args.threads)
    tokens = validation_windows(args.windows)
    length, copyable = earlier_matches(tokens, 512)
    copyable = (length >= 4) & copyable
    bits, names = paired_bits(args.checkpoints, tokens, args.averaged, torch.device(args.device))
    _, start, place = word_places(tokens)
    # The words that started before each target.
    seen = start.cumsum(1)[:, :-1]
    first, later = place[:, 1:] == 1, place[:, 1:] >= 2
    row, rows = paired_rows(bits)
    row("all", np.ones_like(copyable))
    row("copyable (match >= 4)", copyable)
    row("not copyable", ~copyable)
    row("first letters", first)
    row("later letters", later)
    row("other bytes", ~first & ~later)
    for name, mask in (("first letters", first), ("later letters", later)):
        for low, high in VIEWS:
            label = f"{low}" + ("" if high == low else "+" if high > 999 else f"-{high}")
            row(f"{name}, {label} words in view", mask & ~copyable & (seen >= low) & (seen <= high))
    write_rows(args.json, names, args, rows)


if __name__ == "__main__":
    main()
