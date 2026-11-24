"""The gains that a stack's skip fibres learned (--stack-skips), read from the weights alone.

A skip fibre brings a level's read or cells the stream as it stood at the input or after an earlier level, through a
gain for every channel. The gains are whole sixteenths, start at zero and step with the optimizer; a gain that is
still zero brings nothing. For every part that has fibres (a read, or its probes, tags and payloads each, and the
cells) and every place a fibre comes from, this reports the share of channels whose gain is not zero, the mean size
of the gains and their mean with its sign, in sixteenths, and the largest. It sums them by part and by how many levels
a fibre reaches back, and gives the mean reach of a channel: the stream's own count and the summed sizes of its gains,
in a stream neuron's counts. The sums also give the mean size of the gains as they are stored, before they are
rounded to whole sixteenths: gains that are still zero but on their way show there.

In a looped stack every pass is a level of its own here, as it has its own gains; a row names the stack's level too.

usage: skip_gains.py --checkpoint run/model.pt.averaged [more ...] --json skip_gains.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output  # noqa: E402
from snnformer.event_lm import _SKIP_UNITS, EventLM, _skip_gains  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402

_PARTS = ("probes", "tags", "payloads")


def _summary(gains, stored):
    """Of some gains in sixteenths: how many, the share that is not zero, their mean size and their largest, and
    the mean size of the values they are rounded from."""
    sizes = torch.cat([gain.abs().flatten() for gain in gains])
    unrounded = torch.cat([gain.abs().flatten() for gain in stored]) * _SKIP_UNITS
    return {"gains": sizes.numel(), "share_not_zero": float((sizes != 0).float().mean()),
            "mean_size_sixteenths": float(sizes.mean()), "largest_sixteenths": float(sizes.max()),
            "mean_stored_size_sixteenths": float(unrounded.mean())}


@torch.no_grad()
def skip_gains(model):
    """The learned gains of ``model``'s skip fibres: a row for every part and source, and their sums."""
    if model.skips is None:
        raise ValueError("the stack has no skip fibres")
    cfg = model.cfg
    rows, by_part, by_back, everything, reaches = [], defaultdict(list), defaultdict(list), [], {}
    for name, parameter in model.skips.items():
        cells = name.startswith("cells")
        level = int(name[len("cells" if cells else "read"):])
        mixes = 3 if cfg.stack_skips == "parts" and not cells else 1
        # As the forward pass takes them: whole sixteenths, a channel's gains to one mix shrunk to the largest sum.
        stored = parameter.view(mixes, -1, cfg.dim)
        gains = _skip_gains(stored)
        for mix in range(mixes):
            part = "cells" if cells else (_PARTS[mix] if mixes == 3 else "read")
            reaches[f"level {level} {part}"] = float(1 + gains[mix].abs().sum(0).mean() / _SKIP_UNITS)
            for source in range(gains.shape[1]):
                gain = gains[mix, source]
                rows.append({"level": level, "stack_level": model.visits[level], "part": part,
                             "from": "the input" if source == 0 else f"after level {source - 1}",
                             "levels_back": level - source,
                             "share_not_zero": float((gain != 0).float().mean()),
                             "mean_size_sixteenths": float(gain.abs().mean()),
                             "mean_gain_sixteenths": float(gain.mean()),
                             "largest_sixteenths": float(gain.abs().max())})
                pair = (gain, stored[mix, source])
                by_part[part].append(pair)
                by_back[level - source].append(pair)
                everything.append(pair)
    return {"skips": cfg.stack_skips, "whole": _summary(*zip(*everything)),
            "by_part": {part: _summary(*zip(*pairs)) for part, pairs in by_part.items()},
            "by_levels_back": {str(back): _summary(*zip(*pairs)) for back, pairs in sorted(by_back.items())},
            "mean_reach_of_a_channel": reaches, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    record = {"gain_unit": f"1/{_SKIP_UNITS} of a spike", "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, torch.device("cpu"))
        if not isinstance(model, EventLM) or model.skips is None:
            parser.error(f"{path}: the stack has no skip fibres")
        found = skip_gains(model)
        record["runs"][path.parent.name] = {"run_name": checkpoint.get("run_name"), "weights": path.name,
                                            "peak_rate": checkpoint["args"].get("lr"),
                                            "schedule_updates": checkpoint["args"].get("steps"), **found}
        whole = found["whole"]
        print(path.parent.name, f"{whole['gains']:,} gains, {whole['share_not_zero']:.1%} not zero, mean size "
              f"{whole['mean_size_sixteenths']:.3f} sixteenths (stored {whole['mean_stored_size_sixteenths']:.3f}), "
              f"largest {whole['largest_sixteenths']:.0f};",
              " ".join(f"{part} {row['share_not_zero']:.1%}" for part, row in found["by_part"].items()), flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
