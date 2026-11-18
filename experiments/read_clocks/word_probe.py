"""How much a network's stream knows about the words before the one it is completing.

usage: word_probe.py CHECKPOINT... [--windows 1024] [--averaged] [--threads 8] [--device cpu] [--json OUT]
Run from the repository root.

The bytes looked at are those inside a word whose next byte continues the word, with at least six earlier
words in the window: the places where a network completes a word and the words before it can tell which word
is likely. At these bytes a linear readout of the stream is fitted to name, for the word in progress (0 back)
and for each of the five words before it, the word's first letter (26 letters and one class for all else) and
the word itself (the 255 most frequent words of the windows without regard to case, and one class for all
others). The readout is fitted on two windows of four, starts from the class frequencies of those windows and
stops, for every class of its own, at the pass that the third window likes best; the fourth window scores it.
Every number is the bits that the stream removes from the uncertainty about the class: the loss of a readout
that sees nothing (the class frequencies) minus the loss of the readout of the stream.

A decoder is read before its first state read, after every state read and where the output synapses read it.
A Transformer is read after a quarter, a half and three quarters of its blocks and after the last.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments/read_clocks")
from words_in_view import load_weights, validation_windows, word_places  # noqa: E402

from snnformer.event_lm import EventLM  # noqa: E402

BACK = 5
COMMON = 255
LETTERS = 27


def taps(model):
    """The places at which the stream is read: a name, a module, and whether the module's input is the stream."""
    if isinstance(model, EventLM):
        reads = sorted(int(index) for index in (model.state_reads or {}))
        places = [(f"block {reads[0]}, before the first read", model.blocks[reads[0]], False)] if reads else []
        places += [(f"read after block {index}", model.state_reads[str(index)], False) for index in reads]
        return places + [("where the output reads", model.output, True)]
    depth = len(model.blocks)
    chosen = sorted({max(1, depth // 4), max(1, depth // 2), max(1, 3 * depth // 4), depth})
    return [(f"block {index} of {depth}", model.blocks[index - 1], False) for index in chosen]


def first_letter(byte):
    byte = byte + 32 if 65 <= byte <= 90 else byte
    return byte - 97 if 97 <= byte <= 122 else LETTERS - 1


def word_classes(tokens, inside, fitted):
    """The bytes to read and their classes.

    Returns the mask of the bytes [windows, 512] and, for each of them in row order, the first-letter class and
    the word class of the words 0 to BACK back. The frequent words are counted in the fitted windows only.
    """
    words = []
    for row, line in enumerate(tokens):
        found, column = [], 0
        while column < len(line):
            if not inside[row, column]:
                column += 1
                continue
            end = column
            while end < len(line) and inside[row, end]:
                end += 1
            found.append((column, end, bytes(line[column:end].astype(np.uint8).tolist()).lower()))
            column = end
        words.append(found)
    counts = Counter(word for row in np.flatnonzero(fitted) for _, _, word in words[row])
    common = {word: index for index, (word, _) in enumerate(counts.most_common(COMMON))}
    mask = np.zeros((tokens.shape[0], tokens.shape[1] - 1), dtype=bool)
    letters, names = [], []
    for row, found in enumerate(words):
        for index in range(BACK + 1, len(found)):
            start, end, _ = found[index]
            # The word's bytes whose next byte continues the word. The first word of a window may be cut, so
            # the words looked back at start with the second.
            last = min(end - 1, mask.shape[1])
            if last <= start:
                continue
            mask[row, start:last] = True
            back = [found[index - k][2] for k in range(BACK + 1)]
            letters += [[first_letter(word[0]) for word in back]] * (last - start)
            names += [[common.get(word, COMMON) for word in back]] * (last - start)
    return mask, np.array(letters, dtype=np.int64), np.array(names, dtype=np.int64)


def streams(model, windows, mask):
    """The stream at the masked bytes at every tap, as half-precision rows in the order of the mask."""
    kept = {name: [] for name, _, _ in taps(model)}
    current, seen = {}, {}
    hooks = []
    for name, module, takes_input in taps(model):
        def keep(module, inputs, output, name=name, takes_input=takes_input):
            # A module that runs more than once in a pass is read at its last run.
            stream = inputs[0] if takes_input else output
            seen[name] = stream.detach()[current["mask"]].to(torch.float16).cpu()
        hooks.append(module.register_forward_hook(keep))
    with torch.inference_mode():
        for batch, rows in zip(windows.split(16), torch.from_numpy(mask).split(16)):
            current["mask"] = rows.to(windows.device)
            model(batch[:, :-1])
            for name, stream in seen.items():
                kept[name].append(stream)
    for hook in hooks:
        hook.remove()
    return {name: torch.cat(rows) for name, rows in kept.items()}


def fit(features, classes, sizes, parts, device, epochs):
    """Bits removed from every class's uncertainty by a linear readout of the features, on the scored rows.

    ``parts`` names every row's part: 0 fitted, 1 for choosing where to stop, 2 scored.
    """
    features = features.to(device).float()
    fitted = parts == 0
    mean, spread = features[fitted].mean(0), features[fitted].std(0).clamp_min(1e-3)
    features = (features - mean) / spread
    classes = classes.to(device)
    train, train_classes = features[fitted], classes[fitted]
    readout = torch.nn.Linear(features.shape[1], sum(sizes)).to(device)
    # The readout starts as the one that sees nothing: no weights, and the class frequencies as its biases.
    blind = torch.cat([(torch.bincount(train_classes[:, task], minlength=size).float() + 0.5).log()
                       for task, size in enumerate(sizes)])
    with torch.no_grad():
        readout.weight.zero_()
        readout.bias.copy_(blind - torch.cat([part.logsumexp(0).expand(len(part)) for part in blind.split(sizes)]))
    steps = epochs * math.ceil(len(train) / 4096)
    optimizer = torch.optim.AdamW(readout.parameters(), lr=1e-3, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=steps)
    generator = torch.Generator(device=device).manual_seed(0)

    def losses(rows):
        logits = readout(features[rows]).split(sizes, dim=-1)
        return torch.stack([F.cross_entropy(part, classes[rows][:, task]) for task, part in enumerate(logits)])

    def held():
        with torch.inference_mode():
            return losses(parts == 1), losses(parts == 2)

    chosen, scored = held()
    start = scored.clone()
    for _ in range(epochs):
        for rows in torch.randperm(len(train), device=device, generator=generator).split(4096):
            logits = readout(train[rows]).split(sizes, dim=-1)
            loss = sum(F.cross_entropy(part, train_classes[rows][:, task]) for task, part in enumerate(logits))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            schedule.step()
        now_chosen, now_scored = held()
        better = now_chosen < chosen
        chosen, scored = torch.where(better, now_chosen, chosen), torch.where(better, now_scored, scored)
    return ((start - scored) / math.log(2)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--windows", type=int, default=1024)
    parser.add_argument("--averaged", action="store_true",
                        help="read the running weight average where a checkpoint stores one")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    tokens = validation_windows(args.windows)
    inside, _, _ = word_places(tokens)
    part = np.maximum(np.arange(len(tokens)) % 4 - 1, 0)
    mask, letters, names = word_classes(tokens, inside, part == 0)
    parts = torch.from_numpy(np.repeat(part, mask.sum(1))).to(device)
    classes = torch.from_numpy(np.concatenate([letters, names], axis=1))
    sizes = [LETTERS] * (BACK + 1) + [COMMON + 1] * (BACK + 1)
    windows = torch.from_numpy(tokens).to(device)
    print(f"{args.windows} windows, {int(mask.sum())} bytes read, {int((parts == 0).sum())} of them fitted; "
          f"bits removed from the uncertainty about the word 0 to {BACK} back")
    record = {"windows": args.windows, "averaged": args.averaged, "bytes": int(mask.sum()), "checkpoints": []}
    for path in args.checkpoints:
        model, took_average = load_weights(path, args.averaged, device)
        name = f"{Path(path).parent.name}/{Path(path).name}"
        print(f"\n{name} | {'averaged' if took_average else 'as stored'}")
        print(f"{'stream':<34} | first letter of the word " + " ".join(f"{k:>5}" for k in range(BACK + 1))
              + " | the word " + " ".join(f"{k:>5}" for k in range(BACK + 1)))
        kept = streams(model, windows, mask)
        del model
        rows = []
        for tap, features in kept.items():
            gained = fit(features, classes, sizes, parts, device, args.epochs)
            rows.append({"stream": tap, "channels": features.shape[1], "first_letter": gained[:BACK + 1],
                         "word": gained[BACK + 1:]})
            print(f"{tap:<34} | {'':<24} " + " ".join(f"{value:5.2f}" for value in gained[:BACK + 1])
                  + " | " + f"{'':<8} " + " ".join(f"{value:5.2f}" for value in gained[BACK + 1:]), flush=True)
        record["checkpoints"].append({"checkpoint": name, "streams": rows})
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
