"""Where a model's heads look, counted in bytes back and in words back.

usage: head_use.py CHECKPOINT [--windows 24] [--averaged] [--json OUT]      (run from the repository root)

For a Transformer the heads are its attention heads, for the spike decoder the heads of its state reads.
Each row is one head at one kind of prediction: a word's first letter, or a later letter of a word. The
columns are the shares of the head's read by the words that started after a cell (0 is the present word
or the gap after the last one), by the bytes between the cell and the present byte, and three summaries:
``cells`` is the number of cells that share the read (the exponential of the entropy), ``top`` the largest
single share and ``ends`` the share on first bytes after a word, where a word is complete. Scored on
evenly spaced windows of the enwik8 validation split, bytes 128 to 511 of each.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "experiments")
sys.path.insert(0, ".")
from snnformer.event_lm import _word_bytes, _word_starts  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

WORDS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-7", 4, 7), ("8-31", 8, 31), ("32+", 32, 10 ** 6))
BYTES = (("self", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-7", 4, 7), ("8-15", 8, 15), ("16-63", 16, 63),
         ("64+", 64, 10 ** 6))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--averaged", action="store_true",
                        help="use the running weight average where a checkpoint stores one")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_checkpoint(Path(args.checkpoint), torch.device("cpu"))
    averaged = checkpoint.get("averaged_parameters") if args.averaged else None
    if averaged is not None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                parameter.copy_(averaged[name].to(parameter))
    _, validation, _ = load_splits(Path("data/enwik8"))
    starts = np.linspace(0, len(validation) - 600, args.windows).astype(int)
    tokens = torch.from_numpy(np.stack([validation[start:start + 512] for start in starts]).astype(np.int64))
    event = hasattr(model, "state_reads")
    layers = list(model.state_reads.items()) if event else list(enumerate(model.blocks))
    seen = {}
    for index, layer in layers:
        module = layer if event else layer.attn
        module.register_forward_pre_hook(lambda _module, inputs, index=index: seen.__setitem__(index, inputs))
    rows = []
    with torch.no_grad():
        model(tokens)
        steps = tokens.shape[1]
        inside = _word_bytes()[tokens]
        started = _word_starts(inside).cumsum(1)
        ages = started[:, :, None] - started[:, None, :]
        distance = (torch.arange(steps)[:, None] - torch.arange(steps))[None].expand(len(tokens), -1, -1)
        # A first letter is predicted at a byte outside a word whose next byte is inside one.
        first, later, ended = (torch.zeros_like(inside) for _ in range(3))
        first[:, :-1] = ~inside[:, :-1] & inside[:, 1:]
        later[:, :-1] = inside[:, :-1] & inside[:, 1:]
        ended[:, 1:] = inside[:, :-1] & ~inside[:, 1:]
        ends = ended[:, None, :].expand(-1, steps, -1)
        warm = torch.arange(steps) >= 128
        causal = torch.ones(steps, steps, dtype=torch.bool).tril()
        for index, layer in layers:
            if event:
                shares, _ = layer.shares(seen[index][0], tokens)
            else:
                states, rope = seen[index]
                query, key, _ = layer.attn._project(states, rope)
                charge = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
                shares = charge.masked_fill(~causal, -torch.inf).softmax(-1)
            for head in range(shares.shape[1]):
                for where, mask in (("first letter", first), ("later letter", later)):
                    at = mask & warm
                    picked = shares[:, head][at]

                    def part(values, low, high):
                        return float((picked * ((values[at] >= low) & (values[at] <= high))).sum(-1).mean())

                    rows.append({
                        "layer": int(index), "head": head, "where": where,
                        "words_back": {name: part(ages, low, high) for name, low, high in WORDS},
                        "bytes_back": {name: part(distance, low, high) for name, low, high in BYTES},
                        "cells": float(torch.special.entr(picked).sum(-1).exp().mean()),
                        "top": float(picked.amax(-1).mean()),
                        "ends": float((picked * ends[at]).sum(-1).mean()),
                    })
    kind = "state read after block" if event else "attention of layer"
    print(f"{args.checkpoint}: {kind}, {args.windows} windows, {int((first & warm).sum())} first letters, "
          f"{int((later & warm).sum())} later letters; {float(ended[:, 128:].float().mean()):.3f} of the bytes "
          "are first bytes after a word")
    print("layer head  where         | words back: " + " ".join(f"{name:>5}" for name, _, _ in WORDS)
          + " | bytes back: " + " ".join(f"{name:>5}" for name, _, _ in BYTES) + " | cells   top  ends")
    for row in rows:
        print(f"{row['layer']:>5} {row['head']:>4}  {row['where']:<13} |             "
              + " ".join(f"{value:5.2f}" for value in row["words_back"].values()) + " |             "
              + " ".join(f"{value:5.2f}" for value in row["bytes_back"].values())
              + f" | {row['cells']:5.1f} {row['top']:5.2f} {row['ends']:5.2f}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "checkpoint": Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).name,
            "heads": "state reads" if event else "attention", "windows": args.windows, "rows": rows,
        }, indent=1) + "\n")


if __name__ == "__main__":
    main()
