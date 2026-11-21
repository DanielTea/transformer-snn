"""Continue a prompt with a stack of reads and cells as one stream.

The stack hears the prompt once and then every byte that it writes, a byte
at a time. The state of its reads and cells is carried from byte to byte, so
no byte is ever read twice, and a new byte costs one step of the stack
whatever the length of the text so far. The spikes are those of one pass over
the whole text, because a stream cut anywhere fires the same spikes.

``--reach`` lets the reads reach further back than they did in training, as
in eval_stream.py. With ``--teach-levels`` the output has its fast synapses:
every byte of the prompt teaches them, and so does every byte that the stack
writes itself.

With ``--temperature 0`` the most likely byte is written. Otherwise a byte is
drawn from the stack's distribution at that temperature, with ``--seed``. A
model without carried state, as the Transformer or a decoder of the earlier
releases, writes with generate_windows.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

from common import resolve_device
from eval_stream import lengthen_reach
from snnformer.event_lm import EventLM, EventLMConfig
from train_event_lm import load_checkpoint, scoring_logits


@torch.inference_mode()
def continue_stream(model, prompt, count, *, temperature=0.0, generator=None, device="cpu"):
    """The ``count`` bytes that follow ``prompt``, each written from the state that the stream before it left."""
    resolution = model.cfg.resolution
    heard = torch.tensor([list(prompt)], dtype=torch.long, device=device)
    written, state = [], None
    while True:
        counts, state = model.stream(heard, state)
        if len(written) == count:
            return bytes(written)
        logits = scoring_logits(counts, True, resolution)[0, -1]
        if temperature == 0:
            byte = int(logits.argmax())
        else:
            chances = (logits.double().cpu() / temperature).softmax(-1)
            byte = int(torch.multinomial(chances, 1, generator=generator))
        written.append(byte)
        heard = torch.tensor([[byte]], dtype=torch.long, device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompt-file", type=Path)
    parser.add_argument("--bytes", type=int, default=256, help="number of bytes to write")
    parser.add_argument("--temperature", type=float, default=0.8, help="0 writes the most likely byte")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reach", type=int, help="cells that every read reaches back, if more than in training")
    parser.add_argument("--reach-levels", type=int, nargs="+",
                        help="lengthen the reach of the reads after these levels only")
    parser.add_argument("--teach-levels", type=int, nargs="+",
                        help="levels whose cells the fast output synapses hear, which every byte teaches")
    parser.add_argument("--teach-threshold", type=int, help="units of fast charge to one spike of an output neuron")
    parser.add_argument("--teach-budget", type=int, default=EventLMConfig.teach_budget,
                        help="teaching spikes of every byte")
    parser.add_argument("--teach-cascade", type=int, default=EventLMConfig.teach_cascade, metavar="STAGES",
                        help="stages of charge in every fast output synapse, which consolidates itself")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.bytes < 0 or args.temperature < 0:
        parser.error("the output length and the temperature must not be negative")
    if args.reach_levels is not None and args.reach is None:
        parser.error("--reach-levels names the levels that take --reach")
    if (args.teach_levels is None) != (args.teach_threshold is None):
        parser.error("--teach-levels and --teach-threshold go together")
    if args.teach_cascade != 1 and args.teach_levels is None:
        parser.error("--teach-cascade gives stages to the fast synapses of --teach-levels")
    torch.set_num_threads(args.threads)
    try:
        prompt = args.prompt.encode("utf-8") if args.prompt_file is None else args.prompt_file.read_bytes()
        device = resolve_device(args.device)
        model, _ = load_checkpoint(args.checkpoint, device)
        if not isinstance(model, EventLM) or not model.cfg.stack:
            raise ValueError(f"{args.checkpoint}: only a stack of reads and cells writes as a stream")
        if args.reach is not None:
            lengthen_reach(model, args.reach, args.reach_levels)
        if args.teach_levels is not None:
            model.teach(args.teach_levels, args.teach_threshold, args.teach_budget, args.teach_cascade)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not prompt:
        parser.error("the prompt is empty")
    written = continue_stream(model, prompt, args.bytes, temperature=args.temperature,
                              generator=torch.Generator().manual_seed(args.seed), device=device)
    sys.stdout.buffer.write(written)


if __name__ == "__main__":
    main()
