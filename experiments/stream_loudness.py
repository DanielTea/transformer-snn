"""How loud a stream's loudest channels are, where a trained model's norms or pools hear it.

For every token of the stream that a norm (a Transformer's LayerNorm or pool control) or a stack's inhibitory pool
hears, with its d channels x, its summed activity L1 = sum |x|, its energy Q = sum x^2 and its k active channels
(x != 0):

  ratio     sqrt(d Q) / L1: how much more an energy pool hears than a pool of summed activity, which is also how much
            the energy of a stream divided by its mean absolute activity exceeds the energy of a stream divided by
            its root mean square;
  loudness  sqrt(k Q) / L1: 1 when the active channels are equally loud, more when a few are louder than the rest;
  active    k / d: the share of channels that are active.

ratio = loudness * sqrt(d / k): the ratio spreads over tokens with the loudness or with the share of active channels.
Silent tokens (L1 = 0) are counted and left out. A Transformer is read in windows of --window bytes (pre-hooks on
every norm); a stack as --streams unbroken streams from an empty state, in pieces of --piece bytes (the stack's
pools hear the stream through event_lm._heard_activity, which is wrapped while it runs; a looped stack's tied pools
are one place). The quantiles of each measure over tokens go to --json for every place.

usage: python experiments/stream_loudness.py --checkpoint A.pt [B.pt ...] --enwik8 data/enwik8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import _atomic_output
from snnformer import event_lm
from snnformer.event_lm import EventLM
from snnformer.transformer_reference import _PoolNorm
from train_event_lm import load_checkpoint
from train_lm import load_splits

QUANTILES = (0.1, 0.5, 0.9, 0.99)


def loudness(x: torch.Tensor):
    """The ratio, loudness and share of active channels of every token of ``x`` [..., d] as flat float64 arrays,
    with the number of silent tokens, which are left out."""
    x = x.detach().reshape(-1, x.shape[-1]).double()
    summed = x.abs().sum(-1)
    energy = x.square().sum(-1)
    active = (x != 0).sum(-1).double()
    kept = summed > 0
    summed, energy, active = summed[kept], energy[kept], active[kept]
    width = x.shape[-1]
    ratio = (width * energy).sqrt() / summed
    loud = (active * energy).sqrt() / summed
    return ratio.numpy(), loud.numpy(), (active / width).numpy(), int((~kept).sum())


class Places:
    """The measures of every place, gathered over calls."""

    def __init__(self):
        self.parts: dict[str, list] = {}
        self.silent: dict[str, int] = {}
        self.widths: dict[str, int] = {}

    def add(self, place: str, x: torch.Tensor):
        ratio, loud, active, silent = loudness(x)
        self.parts.setdefault(place, []).append((ratio, loud, active))
        self.silent[place] = self.silent.get(place, 0) + silent
        self.widths[place] = int(x.shape[-1])

    def summary(self):
        out = {}
        for place, parts in self.parts.items():
            measures = {}
            for i, name in enumerate(("ratio", "loudness", "active")):
                values = np.concatenate([p[i] for p in parts])
                q = np.quantile(values, QUANTILES) if values.size else [float("nan")] * len(QUANTILES)
                measures[name] = {f"q{round(level * 100)}": round(float(v), 4) for level, v in zip(QUANTILES, q)}
                measures[name]["q99_over_q50"] = round(float(q[-1] / q[1]), 4) if values.size else None
            tokens = int(sum(p[0].size for p in parts))
            out[place] = {"width": self.widths[place], "tokens": tokens, "silent_tokens": self.silent[place],
                          **measures}
        return out


def transformer_places(model: nn.Module, places: Places):
    """Pre-hooks that hand every norm's input to ``places``; returns the handles."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, _PoolNorm)):
            handles.append(module.register_forward_pre_hook(
                lambda module, inputs, name=name: places.add(name, inputs[0])))
    return handles


class StackPlaces:
    """While active, every call of the stack's pool activity hands the stream it hears to ``places``, named by the
    module whose method called it ("stack output" for the output neurons' pool, which the stack itself calls)."""

    def __init__(self, model: EventLM, places: Places):
        self.names = {id(module): (name or "stack output") for name, module in model.named_modules()}
        self.places, self.original = places, None

    def __enter__(self):
        self.original = event_lm._heard_activity

        def heard(spikes, energy=False):
            caller = sys._getframe(1).f_locals.get("self")
            self.places.add(self.names.get(id(caller), type(caller).__name__), spikes)
            return self.original(spikes, energy)

        event_lm._heard_activity = heard
        return self

    def __exit__(self, *exc):
        event_lm._heard_activity = self.original


@torch.no_grad()
def measure_transformer(model, data: torch.Tensor, window: int, batch: int = 16):
    places = Places()
    handles = transformer_places(model, places)
    try:
        usable = (data.numel() // window) * window
        windows = data[:usable].long().view(-1, window)
        for start in range(0, windows.shape[0], batch):
            model(windows[start:start + batch])
    finally:
        for handle in handles:
            handle.remove()
    return places


@torch.no_grad()
def measure_stack(model: EventLM, data: torch.Tensor, streams: int, piece: int):
    places = Places()
    length = data.numel() // streams
    rows = data[:streams * length].long().view(streams, length)
    with StackPlaces(model, places):
        state = None
        for start in range(0, length, piece):
            _, state = model.stream(rows[:, start:start + piece], state)
    return places


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--transformer-bytes", type=int, default=65536)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--stack-bytes", type=int, default=16384)
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--piece", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    torch.set_num_threads(args.threads)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))
    record = {"split": args.split, "quantiles_over_tokens": list(QUANTILES), "checkpoints": []}
    for path in args.checkpoint:
        model, _ = load_checkpoint(path, torch.device("cpu"))
        if isinstance(model, EventLM):
            places = measure_stack(model, data[:args.stack_bytes], args.streams, args.piece)
            form = {"form": "stack", "bytes": args.stack_bytes, "streams": args.streams, "piece": args.piece}
        else:
            places = measure_transformer(model, data[:args.transformer_bytes], args.window)
            form = {"form": "transformer", "bytes": args.transformer_bytes, "window": args.window}
        record["checkpoints"].append({"checkpoint": path.name if path.parent.name == "" else
                                      f"{path.parent.name}/{path.name}", **form, "places": places.summary()})
        print(f"{path.parent.name}: {len(places.parts)} places", flush=True)
    with _atomic_output(args.json, "w", False) as f:
        f.write(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
