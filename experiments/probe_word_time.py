"""Can clock cells learn where words start? A made-up language in which a word depends on the word before last.

Words come from a small dictionary and are separated by one space. With chance
one half a word is the fixed partner of the word before last, otherwise it is
drawn at random. Telling which word comes next therefore needs the word before
last, which lies a varying number of bytes back: no fixed delay reaches it, and
a fibre whose delay counts words does. One layer of race cells on a fixed
random embedding of the bytes and a linear readout are trained on the next
byte. Scored are the bits per first letter of a word, between the entropy
without the word before last and the entropy with it.

The arms differ in what a cluster's fibres bring: plain delays, beat fibres on
a clock that beats at every byte or at every word start, or beat fibres whose
clock cells learn when to beat. For learned clocks the record also says where
they beat. In this language a clock cell hears only the present byte, so the
boundary it can find is the space.

The ``read`` arms train a state read in place of the cells, or in front of
them: a read that knows a cell's age in bytes, in words of the fixed word
clock, or in beats of its heads' own clock cells. Their records list each
head's offsets for the ages in beats and where its clock cell beats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from common import _atomic_output
from snnformer.event_lm import EventLMConfig, _GranuleLayer, _StateRead

LETTERS, SPACE = list(b"abcdefghijklmnop"), 32
ARMS = {
    "delays 0-3": dict(granule_delays=4),
    "delays 0-7": dict(granule_delays=8),
    "every-byte clock": dict(granule_clock="every"),
    "word clock, sum": dict(granule_clock="word", granule_hold="sum"),
    "word clock, mean": dict(granule_clock="word", granule_hold="mean"),
    "word clock, last": dict(granule_clock="word", granule_hold="last"),
    "clock cells, sum": dict(granule_clock="learned", granule_hold="sum"),
    "clock cells, mean": dict(granule_clock="learned", granule_hold="mean"),
    "clock cells, last": dict(granule_clock="learned", granule_hold="last"),
    "read, ages in bytes": dict(read=True),
    "read, word clock": dict(read=True, read_clock="word"),
    "read, word clock, own ages": dict(read=True, read_clock="word", read_beat_start=1.0),
    "read, clock at word ends": dict(read=True, read_clock="end"),
    "read, clock at word ends, own ages": dict(read=True, read_clock="end", read_beat_start=1.0),
    "read, clock cells": dict(read=True, read_clock="learned"),
    "read, clock cells, own ages": dict(read=True, read_clock="learned", read_beat_start=1.0),
    "read, ages in bytes, then cells": dict(read=True, then_cells=True),
    "read, word clock, then cells": dict(read=True, read_clock="word", read_beat_start=1.0, then_cells=True),
    "read, clock at word ends, then cells": dict(read=True, read_clock="end", read_beat_start=1.0, then_cells=True),
    "read, clock cells, then cells": dict(read=True, read_clock="learned", read_beat_start=1.0, then_cells=True),
}
# The state read of the read arms: four heads over the last 32 bytes.
READ = dict(read_after=(0,), read_width=32, read_heads=4, read_window=32, read_mode="race", read_budget=64,
            read_resolution=4)


def language(words, generator):
    """A dictionary of words with distinct letters-in-any-order, and every word's partner."""
    dictionary, bags = [], set()
    while len(dictionary) < words:
        size = int(torch.randint(2, 7, (1,), generator=generator))
        word = [LETTERS[int(index)] for index in torch.randint(0, len(LETTERS), (size,), generator=generator)]
        if tuple(sorted(word)) not in bags:
            bags.add(tuple(sorted(word)))
            dictionary.append(word)
    return dictionary, torch.randperm(words, generator=generator).tolist()


def entropy(chances):
    return -sum(chance * math.log2(chance) for chance in chances if chance > 0)


def bounds(dictionary, partner):
    """Bits per first letter without the word before last and with it."""
    words = len(dictionary)
    firsts = sorted({word[0] for word in dictionary})

    def letters(chance_of):
        return [sum(chance_of(index) for index in range(words) if dictionary[index][0] == first) for first in firsts]

    without = entropy(letters(lambda index: 1 / words))
    known = sum(entropy(letters(lambda index, before=before: 0.5 / words + 0.5 * (index == partner[before])))
                for before in range(words)) / words
    return without, known


def text(rows, length, dictionary, partner, generator):
    """Byte rows, their next bytes, and which next bytes are first letters that the word before last decides."""
    tokens = torch.zeros(rows, length + 1, dtype=torch.long)
    decided = torch.zeros(rows, length + 1, dtype=torch.bool)
    for row in range(rows):
        words, out, marks = [], [], []
        while len(out) < length + 1:
            follows = len(words) >= 2 and bool(torch.rand((), generator=generator) < 0.5)
            index = partner[words[-2]] if follows else int(torch.randint(0, len(dictionary), (1,), generator=generator))
            words.append(index)
            marks += [len(words) > 2] + [False] * len(dictionary[index])
            out += dictionary[index] + [SPACE]
        tokens[row], decided[row] = torch.tensor(out[:length + 1]), torch.tensor(marks[:length + 1])
    return tokens[:, :-1], tokens[:, 1:], decided[:, 1:]


def run(name, args, dictionary, partner, embedding):
    torch.manual_seed(args.seed)
    settings = dict(ARMS[name])
    reading, then_cells = settings.pop("read", False), settings.pop("then_cells", False)
    if "granule_clock" in settings:
        settings.update(granule_beats=args.beats, granule_clock_rate=args.clock_rate)
    if "read_clock" in settings:
        settings.update(read_beats=args.beats, read_clock_rate=args.clock_rate)
    cfg = EventLMConfig(dim=embedding.shape[1], resolution=16, router_charge_bits=8, granule_after=(0,),
                        granule_clusters=8, granule_cells=48, granule_fan=32, granule_budget=16, granule_gain=1.0,
                        **(READ if reading else {}), **settings)
    read = _StateRead(cfg) if reading else None
    layer = _GranuleLayer(cfg, 0) if then_cells or not reading else None
    head = torch.nn.Linear(embedding.shape[1], 256)
    with torch.no_grad():
        for part, synapses in ((read, "project.weight"), (layer, "send")):
            if part is not None:
                part.get_parameter(synapses).normal_(std=0.02)
    parts = [part for part in (read, layer, head) if part is not None]
    optimizer = torch.optim.Adam([parameter for part in parts for parameter in part.parameters()], lr=args.lr)
    data = torch.Generator().manual_seed(args.seed + 1)

    def predict(tokens):
        spikes = embedding[tokens]
        if read is not None:
            spikes = read(spikes, None, tokens)
        if layer is not None:
            spikes = layer(spikes, tokens)
        return head(spikes / cfg.resolution)

    for _ in range(args.steps):
        tokens, targets, _ = text(args.batch, args.length, dictionary, partner, data)
        optimizer.zero_grad()
        F.cross_entropy(predict(tokens).flatten(0, 1), targets.flatten()).backward()
        optimizer.step()
    tokens, targets, decided = text(256, args.length, dictionary, partner, torch.Generator().manual_seed(99))
    with torch.no_grad():
        bits = F.cross_entropy(predict(tokens).flatten(0, 1), targets.flatten(), reduction="none").view(targets.shape)
        bits = bits / math.log(2)
        record = {"arm": name, "first_letter_bits": bits[decided].mean().item(), "bits_per_byte": bits.mean().item()}
        space = tokens == SPACE
        if read is not None and read.beats:
            record["heads"] = [{"offsets_by_age_in_beats": offsets.tolist()} for offsets in read.beat_age * read.rate]
            if read.clock is not None:
                for row, mine in zip(record["heads"], read.clock_beats(embedding[tokens])[0].unbind(-1), strict=True):
                    row.update(beats_per_byte=mine.mean().item(), beats_per_space=mine[space].mean().item(),
                               share_of_beats_on_spaces=(mine[space].sum() / mine.sum().clamp_min(1e-9)).item())
                record["share_of_bytes_that_are_spaces"] = space.float().mean().item()
        if layer is not None and layer.clock is not None:
            # The share of a cluster's clock cells that beat at each byte.
            beating = layer.beats(embedding[tokens], tokens)[0].mean(-1)
            record["clusters"] = [{
                "beats_per_byte": mine.mean().item(),
                "share_of_beats_on_spaces": (mine[space].sum() / mine.sum().clamp_min(1e-9)).item(),
                "beats_per_space": mine[space].mean().item(),
                "share_of_bytes_with_split_cells": ((mine > 0.1) & (mine < 0.9)).float().mean().item(),
            } for mine in beating.unbind(-1)]
            record["share_of_bytes_that_are_spaces"] = space.float().mean().item()
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--words", type=int, default=24)
    parser.add_argument("--beats", type=int, default=3,
                        help="stretches a beat fibre's chain holds, and ages in beats that a read tells apart")
    parser.add_argument("--clock-rate", type=float, default=EventLMConfig.granule_clock_rate)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    generator = torch.Generator().manual_seed(1234)
    dictionary, partner = language(args.words, generator)
    embedding = torch.randint(-16, 17, (256, args.dim), generator=generator).float()
    without, known = bounds(dictionary, partner)
    print(f"{args.words} words, {args.steps} updates of {args.batch} x {args.length} bytes, seed {args.seed}. "
          f"Bits per first letter: {without:.3f} without the word before last, {known:.3f} with it.", flush=True)
    records = []
    for name in args.arms:
        record = run(name, args, dictionary, partner, embedding)
        records.append(record)
        line = f"{name:<31} first letters {record['first_letter_bits']:.3f}  all bytes {record['bits_per_byte']:.3f}"
        for clocks in ("clusters", "heads"):
            if "beats_per_space" in record.get(clocks, [{}])[0]:
                line += "  beats per space " + " ".join(f"{c['beats_per_space']:.2f}" for c in record[clocks])
                line += "  share of beats on spaces " + " ".join(f"{c['share_of_beats_on_spaces']:.2f}"
                                                                 for c in record[clocks])
        print(line, flush=True)
    if args.output is not None:
        result = {"words": [bytes(word).decode() for word in dictionary], "partner": partner,
                  "first_letter_bits_without_the_word_before_last": without,
                  "first_letter_bits_with_the_word_before_last": known,
                  "settings": {key: value for key, value in vars(args).items() if key not in ("output", "arms")},
                  "arms": records}
        with _atomic_output(args.output, "w", overwrite=True) as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
