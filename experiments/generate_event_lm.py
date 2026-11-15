"""Greedy byte generation using the integer spike decoder."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

import torch

from eval_event_lm import load_checkpoint
from snnformer.event_lm import EventLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--bytes", type=int, default=128, help="number of output bytes")
    parser.add_argument("--compact-delays", action="store_true", help="use per-delay rings with the saved weights")
    parser.add_argument("--output-reference", action="store_true",
                        help="subtract a shared firing reference while preserving vocabulary margins")
    args = parser.parse_args()
    if args.bytes < 0:
        parser.error("output length must be nonnegative")
    try:
        prompt = args.prompt.encode("utf-8") if args.prompt_file is None else args.prompt_file.read_bytes()
        model, _ = load_checkpoint(args.checkpoint, torch.device("cpu"))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not prompt or not isinstance(model, EventLM):
        parser.error("a nonempty prompt and an EventLM checkpoint are required")

    try:
        if args.compact_delays or args.output_reference:
            cfg = replace(model.cfg, compact_delays=model.cfg.compact_delays or args.compact_delays,
                          output_reference=model.cfg.output_reference or args.output_reference)
            decoder = EventLM(cfg).eval()
            decoder.load_state_dict(model.state_dict())
            model = decoder
        model = model.to_integer()
    except ValueError as exc:
        parser.error(str(exc))
    state = None
    for token in torch.tensor(list(prompt), dtype=torch.long).split(1):
        counts, state = model.step(token, state)
    result = bytearray()
    for _ in range(args.bytes):
        token = model.byte(counts)
        result.append(int(token.item()))
        counts, state = model.step(token, state)
    sys.stdout.buffer.write(result)


if __name__ == "__main__":
    main()
