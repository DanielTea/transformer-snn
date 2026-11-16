"""Measure writes and the age of recalled charge on native enwik8 decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common import _atomic_output
from eval_event_lm import load_checkpoint
from snnformer.correction_memory import _interval_scan
from snnformer.event_lm import EventLM
from train_lm import ENWIK8_SHA1, load_splits


@torch.inference_mode()
def profile(model, tokens):
    records = []

    def observe(trace, inputs, options, output):
        content, steering, initial, _ = inputs
        radius = options.get("write_radius", steering)
        lower, upper = trace._bounds(content, radius)
        initial = torch.zeros_like(content[:, 0]) if initial is None else initial
        after = _interval_scan(lower, upper, initial)
        before = torch.cat((initial[:, None], after[:, :-1]), dim=1)
        changed = after != before
        silent = steering == 0
        silent_with_charge = silent & (before != 0)
        positions = torch.arange(1, tokens.shape[1] + 1, device=tokens.device)[None, :, None]
        writes = torch.where(changed, positions, 0).cummax(1).values
        latest = torch.cat((torch.zeros_like(writes[:, :1]), writes[:, :-1]), dim=1)
        ages = positions - latest
        pulses = output[0].abs()
        groups = (("1-7", 1, 8), ("8-31", 8, 32), ("32-127", 32, 128), ("128+", 128, tokens.shape[1] + 1))
        age_bins = {label: int(pulses[(ages >= low) & (ages < high)].sum()) for label, low, high in groups}
        records.append({
            "block": len(records) + 1,
            "channel_bytes": content.numel(),
            "changed_cells": int(changed.sum()),
            "closed_aperture_cells_with_charge": int(silent_with_charge.sum()),
            "writes_to_charged_cells_with_closed_aperture": int((silent_with_charge & changed).sum()),
            "write_protection_pulses": int((radius - steering).sum()),
            "coincident_read_pulses_by_bytes_since_last_change": age_bins,
        })

    hooks = [block.trace.register_forward_hook(observe, with_kwargs=True) for block in model.blocks]
    try:
        model(tokens)
    finally:
        for hook in hooks:
            hook.remove()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=4096)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if not isinstance(model, EventLM) or not model.cfg.local_trace:
        parser.error("a held-charge checkpoint is required")
    _, validation, test = load_splits(args.enwik8)
    split = validation if args.split == "validation" else test
    if not 0 < args.bytes <= len(split):
        parser.error("sample must contain at least one byte and fit within the split")
    sample = split[:args.bytes]
    tokens = torch.from_numpy(sample.copy()).long()[None]
    record = {
        "checkpoint": str(args.checkpoint),
        "dataset": "enwik8",
        "data_sha1": ENWIK8_SHA1,
        "split": args.split,
        "input_bytes": args.bytes,
        "input_sha256": hashlib.sha256(sample).hexdigest(),
        "blocks": profile(model.to_integer(), tokens),
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
