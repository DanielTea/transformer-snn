"""Continue a prompt with a model that reads windows.

A decoder with state reads or cell layers has no carried state: it reads a
window from an empty start. So every new byte runs the last ``--context``
bytes again, which costs a whole forward pass per byte and needs nothing but
the saved weights. Each prediction has the bytes before it that the model
would have in a warmed window. This works for any model that
train_event_lm.py saves, the Transformer included.

With ``--temperature 0`` the most likely byte is written. Otherwise a byte is
drawn from the model's distribution at that temperature, with ``--seed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from common import resolve_device
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits


@torch.inference_mode()
def continue_bytes(model, prompt, count, *, context, temperature=0.0, generator=None, device="cpu"):
    """The ``count`` bytes that follow ``prompt``, each predicted from the last ``context`` bytes."""
    event = isinstance(model, EventLM)
    resolution = model.cfg.resolution if event else 1
    written = list(prompt)
    for _ in range(count):
        window = torch.tensor([written[-context:]], dtype=torch.long, device=device)
        logits = scoring_logits(model(window), event, resolution)[0, -1]
        if temperature == 0:
            byte = int(logits.argmax())
        else:
            chances = (logits.double() / temperature).softmax(-1).cpu()
            byte = int(torch.multinomial(chances, 1, generator=generator))
        written.append(byte)
    return bytes(written[len(prompt):])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--bytes", type=int, default=256, help="number of bytes to write")
    parser.add_argument("--context", type=int, default=512, help="bytes the model reads for each new byte")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 writes the most likely byte")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.bytes < 0 or args.context < 1 or args.temperature < 0:
        parser.error("the output length and the temperature must not be negative and the context at least one byte")
    torch.set_num_threads(args.threads)
    try:
        prompt = args.prompt.encode("utf-8") if args.prompt_file is None else args.prompt_file.read_bytes()
        device = resolve_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not prompt:
        parser.error("the prompt is empty")
    if args.context > checkpoint["args"]["ctx"]:
        parser.error(f"the model was trained on windows of {checkpoint['args']['ctx']} bytes")
    written = continue_bytes(model, prompt, args.bytes, context=args.context, temperature=args.temperature,
                             generator=torch.Generator().manual_seed(args.seed), device=device)
    sys.stdout.buffer.write(written)


if __name__ == "__main__":
    main()
