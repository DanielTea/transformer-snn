"""Where do a decoder's learned clocks beat?

For every cluster of every granule layer with clock cells, on windows of an
enwik8 split: the beats per byte and clock cell, the share of them that falls
on the first byte of a word, the beats per word start, the beats per byte by
the kind of the byte, and the share of bytes at which the cluster's clock
cells are split, so that a part of its beat fibres moves on. A word is a run
of letters, digits and bytes of longer characters, as the fixed word clock
has it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import _atomic_output, resolve_device
from train_event_lm import load_checkpoint
from train_lm import load_splits

KINDS = (("letter", b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"), ("digit", b"0123456789"),
         ("space", b" "), ("newline", b"\n"), ("punctuation", b".,;:!?'\"()-"))


def text_features(tokens):
    """The kind of every byte, whether it starts a word, and how many word bytes stand right before it."""
    kinds = np.full(256, len(KINDS))
    for index, (_, group) in enumerate(KINDS):
        kinds[list(group)] = index
    # The bytes of longer characters count as letters.
    kinds[128:] = 0
    kind = kinds[tokens]
    inside = kind <= 1
    earlier = np.concatenate([np.zeros((len(kind), 1), dtype=bool), inside[:, :-1]], axis=1)
    run = np.zeros_like(kind)
    for column in range(1, kind.shape[1]):
        run[:, column] = np.where(inside[:, column - 1], run[:, column - 1] + 1, 0)
    return kind, inside & ~earlier, run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--windows", type=int, default=64)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    layers = {name: layer for name, layer in (getattr(model, "granule_layers", None) or {}).items()
              if getattr(layer, "clock", None) is not None}
    if not layers:
        parser.error("the checkpoint has no learned clocks")
    _, validation, test = load_splits(args.data)
    data = np.asarray(validation if args.split == "validation" else test)
    starts = np.linspace(0, len(data) - args.context - 1, args.windows).astype(np.int64)
    tokens = np.stack([data[start:start + args.context] for start in starts]).astype(np.int64)
    kind, word_start, run = text_features(tokens)
    fired, beating = {}, {name: [] for name in layers}
    for name, layer in layers.items():
        layer.tick.register_forward_hook(lambda _module, _inputs, beats, name=name: fired.__setitem__(name, beats))
    with torch.inference_mode():
        for part in torch.from_numpy(tokens).split(args.batch):
            model(part.to(device))
            for name in layers:
                # The share of a cluster's clock cells that beat at each byte.
                beating[name].append(fired[name].float().mean(-1).cpu().numpy())
    names = [name for name, _ in KINDS] + ["markup"]
    letters = kind == 0
    result = {"checkpoint": args.checkpoint.name, "steps": checkpoint["args"]["steps"], "split": args.split,
              "bytes": int(tokens.size), "share_of_bytes_that_start_a_word": float(word_start.mean()), "layers": {}}
    print(f"{args.checkpoint}: {tokens.size} {args.split} bytes, {100 * word_start.mean():.1f}% of them start a word")
    print("cluster: beats per byte and cell | share of beats on a word start | beats per word start | split cells | "
          + ", ".join(names) + " | letter after 1, 2, 3+ word bytes")
    for name, parts in beating.items():
        beats = np.concatenate(parts)
        print(f"layer after block {name}")
        rows = []
        for cluster in range(beats.shape[-1]):
            mine = beats[..., cluster]
            row = {
                "beats_per_byte": float(mine.mean()),
                "share_of_beats_on_word_starts": float(mine[word_start].sum() / max(mine.sum(), 1e-9)),
                "beats_per_word_start": float(mine[word_start].mean()),
                "share_of_bytes_with_split_cells": float(((mine > 0.1) & (mine < 0.9)).mean()),
                "beats_per_byte_by_kind": {label: float(mine[kind == index].mean())
                                           for index, label in enumerate(names) if (kind == index).any()},
                "beats_per_letter_after_word_bytes": {label: float(mine[letters & pick].mean()) for label, pick in
                                                      (("1", run == 1), ("2", run == 2), ("3+", run >= 3))},
            }
            rows.append(row)
            print(f"  {cluster:>2}: {row['beats_per_byte']:.3f} | {row['share_of_beats_on_word_starts']:.2f} | "
                  f"{row['beats_per_word_start']:.2f} | {row['share_of_bytes_with_split_cells']:.2f} | "
                  + " ".join(f"{value:.2f}" for value in row["beats_per_byte_by_kind"].values()) + " | "
                  + " ".join(f"{value:.2f}" for value in row["beats_per_letter_after_word_bytes"].values()))
        result["layers"][name] = rows
    if args.output is not None:
        with _atomic_output(args.output, "w", overwrite=True) as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
