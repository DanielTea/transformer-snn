"""Measure spare steering for write-driven reads on real validation text."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_spike_code import _IntegerExecution
from common import _atomic_output
from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLM
from train_lm import ENWIK8_SHA1, load_splits


def measure(module, inputs, widths):
    read, steering, present, correction = inputs
    threshold = module.emit.threshold
    aperture = steering.clamp(max=module.emit.cap) * threshold
    drive = module.drive_ticks(present, correction)
    current = read * threshold + drive
    admitted = current.clamp(-aperture, aperture)
    blocked = (current - admitted).abs()
    eligible = blocked.minimum(module.emit.cap * threshold - aperture)
    spare = ((aperture - admitted.abs()) // threshold) * threshold
    closed = steering.eq(0)
    counts = {
        "input_ticks": drive.abs().sum(dtype=torch.int64).item(),
        "blocked_ticks": blocked.sum(dtype=torch.int64).item(),
        "blocked_at_closed_cells_ticks": blocked.masked_fill(~closed, 0).sum(dtype=torch.int64).item(),
        "pair_recoverable_ticks": 0,
        "pair_recoverable_at_closed_cells_ticks": 0,
        "four_cell_upper_bound_ticks": 0,
        "delay_group_upper_bound_ticks": 0,
    }
    for width, demand, stock, silent in zip(widths, eligible.split(widths, -1), spare.split(widths, -1),
                                            closed.split(widths, -1), strict=True):
        paired = width - width % 2
        if paired:
            neighbor = stock[..., :paired].unflatten(-1, (-1, 2)).flip(-1).flatten(-2)
            recovered = demand[..., :paired].minimum(neighbor)
            counts["pair_recoverable_ticks"] += recovered.sum(dtype=torch.int64).item()
            counts["pair_recoverable_at_closed_cells_ticks"] += recovered.masked_fill(
                ~silent[..., :paired], 0,
            ).sum(dtype=torch.int64).item()
        pad = (-width) % 4
        four_cell = [F.pad(values, (0, pad)).unflatten(-1, (-1, 4)).sum(-1, dtype=torch.int64)
                     for values in (demand, stock)]
        counts["four_cell_upper_bound_ticks"] += four_cell[0].minimum(four_cell[1]).sum().item()
        counts["delay_group_upper_bound_ticks"] += demand.sum(-1, dtype=torch.int64).minimum(
            stock.sum(-1, dtype=torch.int64),
        ).sum().item()
    return counts


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if not isinstance(model, EventLM) or model.cfg.trace_read_drive is None:
        parser.error("the checkpoint must have driven held-charge reads")
    _, validation, _ = load_splits(args.enwik8)
    window_bytes = 4096
    starts = np.linspace(0, len(validation) - window_bytes, 4, dtype=np.int64)
    windows = validation[starts[:, None] + np.arange(window_bytes)]

    native = model.to_integer()
    blocks = []
    handles = [block.trace.read_drive.register_forward_hook(
        lambda module, inputs, _output, widths=block.widths: blocks.append(measure(module, inputs, widths)),
    ) for block in native.blocks]
    try:
        with _IntegerExecution():
            native(torch.as_tensor(windows, dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    assert len(blocks) == model.cfg.depth
    with args.checkpoint.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    record = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": checksum,
        "dataset_sha1": ENWIK8_SHA1, "split": "validation", "window_starts": starts.tolist(),
        "window_bytes": window_bytes, "sample_sha256": hashlib.sha256(windows).hexdigest(),
        "by_block": blocks, "totals": {key: sum(row[key] for row in blocks) for key in blocks[0]},
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(record, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
