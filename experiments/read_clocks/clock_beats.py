"""Where the clock cells of a decoder's read heads beat, on validation text.

usage: clock_beats.py CHECKPOINT [--windows 24] [--averaged] [--device cpu] [--json OUT]
Run from the repository root.

With ``--read-clock learned`` every read head has a clock cell, and its spike is the beat by which the head
counts the age of an earlier byte. For every clock cell this prints

  the share of the bytes at which it beats,
  the shares of its beats that fall on the first byte after a word, on a word's first letter, on a later
  letter of a word and on any other byte,
  the share of the first bytes after a word that get a beat,
  and the bytes that take most of its beats, each with its share of them.

A word is a run of letters, digits and bytes above 127, as everywhere in the study.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "experiments/read_clocks")
from words_in_view import load_weights, validation_windows  # noqa: E402

from snnformer.event_lm import _word_bytes, _word_ends, _word_starts  # noqa: E402

PLACES = ("after a word", "first letter", "later letter", "other")


def watch(model):
    """Keep the beats of every read's clock cells while the model runs, as [windows, bytes, heads] per read."""
    seen = {}
    for name, read in model.state_reads.items():
        if read.clock is None:
            continue
        beat = read.clock_beats

        def keep(spikes, tag=None, beat=beat, name=name):
            beats, rates = beat(spikes, tag)
            seen.setdefault(name, []).append(beats.detach().cpu() > 0)
            return beats, rates
        read.clock_beats = keep
    return seen


def shown(byte):
    """A byte as it reads in a table."""
    return {32: "space", 10: "newline", 9: "tab"}.get(byte, chr(byte) if 32 < byte < 127 else f"byte {byte}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--averaged", action="store_true", help="take the running weight average of a snapshot")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    model, took_average = load_weights(args.checkpoint, args.averaged, device)
    model.eval()
    if model.state_reads is None or all(read.clock is None for read in model.state_reads.values()):
        raise SystemExit(f"{args.checkpoint}: no read of this decoder has clock cells")
    tokens = torch.from_numpy(validation_windows(args.windows))[:, :-1]
    seen = watch(model)
    with torch.inference_mode():
        for batch in tokens.split(4):
            model(batch.to(device))
    inside = _word_bytes()[tokens]
    ends, starts = _word_ends(inside), _word_starts(inside)
    places = dict(zip(PLACES, (ends, starts, inside & ~starts, ~inside & ~ends)))
    print(f"{args.checkpoint}: {'averaged weights' if took_average else 'weights as stored'}, {args.windows} "
          f"validation windows of {tokens.shape[1]} bytes; {ends.float().mean():.3f} of the bytes are first bytes "
          f"after a word")
    print(f"{'read after block':>16} {'head':>4} | {'beats per byte':>14} | of its beats: "
          + " ".join(f"{place:>12}" for place in PLACES) + f" | {'word ends with a beat':>21} | bytes with most beats")
    rows = []
    for name, batches in seen.items():
        beats = torch.cat(batches)
        for head in range(beats.shape[-1]):
            mine = beats[..., head]
            total = max(1, int(mine.sum()))
            share = {place: float((mine & mask).sum()) / total for place, mask in places.items()}
            counts = torch.bincount(tokens[mine], minlength=256)
            top = [(int(byte), float(counts[byte]) / total) for byte in counts.argsort(descending=True)[:4]
                   if counts[byte] > 0]
            row = {"read_after": int(name), "head": head, "beats_per_byte": float(mine.float().mean()),
                   "share_of_beats": share, "word_ends_with_a_beat": float(mine[ends].float().mean()),
                   "bytes_with_most_beats": [{"byte": byte, "share": part} for byte, part in top]}
            rows.append(row)
            print(f"{name:>16} {head:>4} | {row['beats_per_byte']:>14.3f} |               "
                  + " ".join(f"{share[place]:>12.2f}" for place in PLACES)
                  + f" | {row['word_ends_with_a_beat']:>21.2f} | "
                  + "  ".join(f"{shown(byte)} {part:.2f}" for byte, part in top))
    if args.json is not None:
        args.json.write_text(json.dumps({"checkpoint": args.checkpoint, "averaged": took_average,
                                         "windows": args.windows, "bytes": int(tokens.numel()),
                                         "first_bytes_after_a_word": float(ends.float().mean()), "rows": rows},
                                        indent=1) + "\n")


if __name__ == "__main__":
    main()
