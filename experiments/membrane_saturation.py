"""How often the membranes of a stack sit at their bound, by kind of neuron, on the first bytes of a split as streams.

A membrane holds the charge of a neuron's largest count at its resting threshold. A charge beyond that bound is cut,
so the neuron fires its largest count whatever more it hears, and through a plain slope it learns nothing from the
part that was cut. v15 found that at a learning rate of 0.001 the fine-pool stack's reads sat at their bound for 5%
of their entries after 12,000 updates (18% at the top level) and almost never at 0.0003 (docs/STREAM_FORM.md). This
reports, for every population of neurons that fires through a membrane with a bound, the share of its entries whose
charge reached the bound (for excitatory cells, from above only). Populations whose pool lifts the ceiling divide the
charge before bounding it and are left out. Streams run as eval_stream.py runs them.

usage: membrane_saturation.py --checkpoint run/model.pt.averaged [more ...] --enwik8 data/enwik8 \\
    --split validation --bytes 65536 --json saturation.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from eval_stream import stream_rows  # noqa: E402
from snnformer.event_lm import EventLM, _ExcitatoryFire, _IntegrateFire  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402


def kind_of(name):
    """The kind and level of a population from its module name: reads, cells, the cells' stream neurons and others."""
    level = re.search(r"\.(\d+)\.", f".{name}.")
    level = int(level.group(1)) if level else None
    if name.startswith("state_reads."):
        return "reads", level
    if name.startswith("granule_layers.") and name.endswith(".fire"):
        return "cells", level
    if name.startswith("granule_layers.") and name.endswith(".emit"):
        return "stream after cells", level
    return "other", level


def saturation(model, inputs, piece, device, skip=0):
    """The share of entries at the membrane bound, per kind of population and per level."""
    tallies = defaultdict(lambda: [0, 0])
    position = [0]
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, _IntegrateFire) or module.lifted or module.dendrite:
            continue
        bound = module.threshold * module.cap

        def record(_module, args, _kwargs, name=name, bound=bound, excitatory=isinstance(module, _ExcitatoryFire)):
            current = args[0].detach()
            if current.ndim < 2:
                return
            current = current[:, max(0, skip - position[0]):]
            if current.numel() == 0:
                return
            cut = current >= bound if excitatory else current.abs() >= bound
            tally = tallies[name]
            tally[0] += current.numel()
            tally[1] += int(cut.sum())

        handles.append(module.register_forward_pre_hook(record, with_kwargs=True))
    try:
        state = None
        with torch.no_grad():
            for start in range(0, inputs.shape[1], piece):
                _, state = model.stream(inputs[:, start:start + piece].to(device), state)
                position[0] = start + piece
    finally:
        for handle in handles:
            handle.remove()
    kinds, levels = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for name, (entries, cut) in tallies.items():
        kind, level = kind_of(name)
        kinds[kind][0] += entries
        kinds[kind][1] += cut
        if level is not None and kind in ("reads", "cells"):
            levels[f"{kind} {level}"][0] += entries
            levels[f"{kind} {level}"][1] += cut
    share = lambda pair: pair[1] / pair[0] if pair[0] else 0.0  # noqa: E731
    return {"by_kind": {kind: share(pair) for kind, pair in sorted(kinds.items())},
            "by_level": {key: share(pair) for key, pair in sorted(levels.items(), key=lambda item: (
                item[0].split()[0], int(item[0].split()[1])))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=65536, help="the first bytes of the split")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=1024, help="unscored bytes that a part after the first hears")
    parser.add_argument("--piece", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))[:args.bytes]
    inputs, _, _ = stream_rows(data, args.streams, args.lead)
    record = {"split": args.split, "first_bytes_of_split": args.bytes, "streams": args.streams, "lead": args.lead,
              "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if not isinstance(model, EventLM) or not model.cfg.stack:
            parser.error(f"{path}: saturation is measured on a stack of reads and cells")
        found = saturation(model, inputs, args.piece, device, skip=args.lead)
        record["runs"][path.parent.name] = {"run_name": checkpoint.get("run_name"),
                                            "peak_rate": checkpoint["args"].get("lr"),
                                            "updates": checkpoint["args"].get("steps"), **found}
        print(path.parent.name, checkpoint["args"].get("lr"),
              " ".join(f"{kind} {value:.4f}" for kind, value in found["by_kind"].items()), flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
