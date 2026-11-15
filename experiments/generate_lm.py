"""Generate text from a temporal spiking byte-language-model checkpoint."""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path

import torch

from common import resolve_device
from probe_lm_context import load_model
from snnformer import TemporalSpikingLM


@torch.inference_mode()
def generate(model, prompt: bytes, byte_count: int, temperature: float, seed: int, device) -> bytes:
    if byte_count == 0:
        return prompt

    state = None
    prefix = torch.tensor(list(prompt), dtype=torch.long, device=device).unsqueeze(1)
    for token in prefix[:-1]:
        _, state = model.step(token, state)
    token = prefix[-1]

    # A CPU generator supports seeded sampling on every model backend, including MPS.
    rng = torch.Generator().manual_seed(seed) if temperature > 0 else None
    result = bytearray(prompt)
    for _ in range(byte_count):
        logits, state = model.step(token, state)
        if rng is None:
            token = logits.argmax(dim=-1)
        else:
            scores = logits.cpu().double()
            scores = (scores - scores.max()) / max(temperature, sys.float_info.min)
            token = torch.multinomial(scores.softmax(dim=-1), 1, generator=rng).squeeze(-1).to(device)
        result.append(token.item())
    return bytes(result)


def display_text(raw: bytes) -> str:
    # Escape terminal controls while preserving tabs, newlines, and valid Unicode text.
    visible = []
    for char in raw.decode("utf-8", errors="replace"):
        code = ord(char)
        if (code < 32 and char not in "\t\n") or 127 <= code <= 159:
            char = f"\\x{code:02x}"
        visible.append(char)
    return "".join(visible)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="byte LM checkpoint from train_lm.py")
    parser.add_argument("--prompt", required=True, help="literal UTF-8 text")
    parser.add_argument("--bytes", type=int, default=64, dest="byte_count", help="number of new bytes (default: 64)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    parser.add_argument("--temperature", type=float, default=0.0, help="sampling temperature (0 for greedy)")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    args = parser.parse_args()

    if not args.prompt:
        parser.error("--prompt must contain at least one byte")
    if args.byte_count < 0:
        parser.error("--bytes must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if not 0 <= args.seed < 2**64:
        parser.error("--seed must be an unsigned 64-bit integer")

    try:
        prompt = args.prompt.encode("utf-8")
    except UnicodeEncodeError:
        parser.error("--prompt must be valid UTF-8 text")
    try:
        device = resolve_device(args.device)
        if device.type == "cpu":
            torch.set_num_threads(1)
        model, name = load_model(args.checkpoint.expanduser(), device)
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, TypeError, AttributeError,
            EOFError, pickle.UnpicklingError) as exc:
        parser.error(str(exc))
    if not isinstance(model, TemporalSpikingLM):
        parser.error(f"{name}: expected a temporal_spike or temporal_spike_reset checkpoint")

    raw = generate(model, prompt, args.byte_count, args.temperature, args.seed, device)
    print(display_text(raw))


if __name__ == "__main__":
    main()
