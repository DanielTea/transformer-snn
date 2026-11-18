"""What does a checkpoint not know that a count table of the training text could tell it?

Each table counts which byte followed one view of the context: the last bytes, the letters of the present word, the
present word and the words before it. A table's counts are mixed into a checkpoint's own prediction (the prediction
stands in for some number of counts, fitted on half of the windows per size of table entry) and the bits saved on
the other half are reported. A table that saves one checkpoint more than another holds what the first is missing.

usage: expert_mix.py SCORES.npz [--long 768] [--least 129]     (run from the repository root; SCORES from context_split.py)
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from train_lm import load_splits  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("scores")
parser.add_argument("--long", type=int, default=768)
parser.add_argument("--least", type=int, default=129, help="fewest bytes in view of a scored target")
args = parser.parse_args()
train, validation, _ = load_splits(Path("data/enwik8"))
train, data = np.asarray(train, dtype=np.uint8), np.array(validation)
scores = np.load(args.scores)
long = scores["source"] == 0
bits = scores["bits"][:, long]
view, run, kind = scores["view"][long], scores["run"][long], scores["kind"][long]
starts = np.linspace(0, len(data) - 512 - 2, args.long).astype(np.int64)
tokens = np.stack([data[start:start + 513] for start in starts]).astype(np.int64)
assert tokens[:, 1:].size == bits.shape[1]
target = tokens[:, 1:].ravel()
window = np.arange(args.long)[:, None].repeat(512, 1).ravel()

letters = np.zeros(257, dtype=bool)
letters[list(b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")] = True
BASE = np.uint64(0x9E3779B97F4A7C15)
INVERSE = np.uint64(pow(0x9E3779B97F4A7C15, -1, 1 << 64))
ODD = np.uint64(0xBF58476D1CE4E5B9)


def mix(a, b):
    with np.errstate(over="ignore"):
        value = (a ^ (b + np.uint64(0x94D049BB133111EB) + (a << np.uint64(6)) + (a >> np.uint64(2)))) * ODD
        return value ^ (value >> np.uint64(29))


def keys(text):
    """For every byte of ``text``, 56-bit keys of the context before it: one (name, keys) pair per view, in turn."""
    count = len(text)
    index = np.arange(count)
    with np.errstate(over="ignore"):
        power = np.cumprod(np.concatenate(([np.uint64(1)], np.full(count, BASE))))
        inverse = np.cumprod(np.concatenate(([np.uint64(1)], np.full(count, INVERSE))))
        prefix = np.concatenate(([np.uint64(0)], np.cumsum((text.astype(np.uint64) + np.uint64(1)) * power[:count])))
        del power

        def stretch(first, end):
            """The hash of text[first:end], the same wherever the stretch stands."""
            return (prefix[end] - prefix[first]) * inverse[first]

        letter = letters[text]
        last_other = np.maximum.accumulate(np.where(~letter, index, -1))
        # The present word starts after the last byte before the target that is no letter.
        start = np.concatenate(([0], last_other[:-1] + 1))
        del last_other
        word = stretch(start, index)
        separator = np.where(start > 0, text[np.maximum(start - 1, 0)].astype(np.uint64), np.uint64(256))
        # A word ends at a byte that is no letter and follows one.
        ends = ~letter & np.concatenate(([False], letter[:-1]))
        last_end = np.maximum.accumulate(np.where(ends, index, -1))
        back = np.concatenate(([-1], last_end[:-1]))
        before = []
        for _ in range(3):
            before.append(np.where(back >= 0, word[np.maximum(back, 0)], np.uint64(1)))
            back = np.where(back > 0, last_end[np.maximum(back - 1, 0)], -1)
        del back, last_end, ends
        value = mix(word, separator)
        yield "present word", value >> np.uint64(8)
        yield ("1 word before, without the present letters",
               mix(mix(separator, before[0]), (index == start).astype(np.uint64)) >> np.uint64(8))
        del word, separator, start
        for words in range(3):
            value = mix(value, before[words])
            yield f"present word + {words + 1} word{'s' if words else ''} before", value >> np.uint64(8)
        del value, before
        for size in (3, 4, 5, 6, 8, 12):
            seen = index >= size
            value = np.where(seen, mix(stretch(np.maximum(index - size, 0), index), np.uint64(size)), np.uint64(0))
            yield f"last {size} bytes", value >> np.uint64(8)


def table(context, following):
    """Counts of every (context, follower) pair, the contexts' totals and their numbers of different followers."""
    pairs, counts = np.unique((context << np.uint64(8)) | following.astype(np.uint64), return_counts=True)
    contexts, first, different = np.unique(pairs >> np.uint64(8), return_index=True, return_counts=True)
    return pairs, counts, contexts, np.add.reduceat(counts, first), different


def look_up(entry, context, following):
    pairs, counts, contexts, totals, different = entry
    place = np.minimum(np.searchsorted(contexts, context), len(contexts) - 1)
    known = contexts[place] == context
    pair = (context << np.uint64(8)) | following.astype(np.uint64)
    spot = np.minimum(np.searchsorted(pairs, pair), len(pairs) - 1)
    hits = np.where(pairs[spot] == pair, counts[spot], 0)
    return np.where(known, totals[place], 0), hits, np.where(known, different[place], 0)


window_keys = [{name: value[1:] for name, value in keys(row.astype(np.uint8))} for row in tokens]
names = list(window_keys[0])
queries = {name: np.concatenate([part[name] for part in window_keys]) for name in names}
print(f"{len(train)} training bytes, {bits.shape[1]} scored targets in {args.long} windows", flush=True)
found = {}
for name, context in keys(train):
    found[name] = look_up(table(context, train), queries[name], target)
    print(f"  table '{name}': {100 * (found[name][0] > 0).mean():.1f}% of the targets' contexts occur in training",
          flush=True)
    del context
# What the window itself has shown: the bytes that followed the same view earlier in the same window.
for name in ("present word", "last 3 bytes"):
    totals, hits, different = (np.zeros(len(target), dtype=np.int64) for _ in range(3))
    for row in range(args.long):
        seen = {}
        base = row * 512
        for column in range(512):
            key, byte = int(queries[name][base + column]), int(target[base + column])
            followers = seen.get(key)
            if followers is None:
                followers = seen[key] = {}
            else:
                totals[base + column] = sum(followers.values())
                hits[base + column] = followers.get(byte, 0)
                different[base + column] = len(followers)
            followers[byte] = followers.get(byte, 0) + 1
    found[f"this window: {name}"] = (totals, hits, different)

scored = view >= args.least
fit, held = scored & (window % 2 == 0), scored & (window % 2 == 1)
sizes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 1 << 62])
stands_for = np.array([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, np.inf])
is_letter = kind < 2
groups = (("all", np.ones_like(is_letter)), ("first letter", is_letter & (run == 0)),
          ("letter after 1-2", is_letter & (run >= 1) & (run <= 2)), ("letter after 3+", is_letter & (run >= 3)),
          ("not a letter", ~is_letter))
print(f"\nBits per target with {args.least}+ bytes in view, on the held-out half: "
      + ", ".join(f"[{i}] {b[held].mean():.4f}" for i, b in enumerate(bits)))
print(f"\nBits saved per target of the group by mixing a table in\n{'table':<46} {'group':<18} "
      + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(len(bits)))
      + " | saved more than [0]: " + " ".join(f"{'[' + str(i) + ']':>8}" for i in range(1, len(bits))))
for name, (totals, hits, different) in found.items():
    bucket = np.searchsorted(sizes, totals, side="right") * 4 + np.minimum(np.searchsorted([2, 3, 5], different,
                                                                                            side="right"), 3)
    saved = []
    for model in bits:
        probability = np.exp2(-model)
        mixed = model.copy()
        for value in np.unique(bucket[totals > 0]):
            inside = (bucket == value) & (totals > 0)
            train_part = inside & fit
            if train_part.sum() < 20:
                continue
            best, best_loss = np.inf, model[train_part].mean()
            for pseudo in stands_for[:-1]:
                loss = -np.log2((hits[train_part] + pseudo * probability[train_part])
                                / (totals[train_part] + pseudo)).mean()
                if loss < best_loss:
                    best, best_loss = pseudo, loss
            if math.isfinite(best):
                mixed[inside] = -np.log2((hits[inside] + best * probability[inside]) / (totals[inside] + best))
        saved.append(model - mixed)
    for label, group in groups:
        here = held & group
        means = [s[here].mean() for s in saved]
        print(f"{name:<46} {label:<18} " + " ".join(f"{m:8.4f}" for m in means) + " |" + " " * 22
              + " ".join(f"{m - means[0]:+8.4f}" for m in means[1:]))
