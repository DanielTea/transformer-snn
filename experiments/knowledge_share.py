"""How much of every state read goes to its knowledge cells: the share of each read's cell spikes that the knowledge
cells take, by level and head, over bytes of the enwik8 validation split read in windows, and how the share moves with
the bytes of the window behind the read (a read at the start of a window has few cells of the stream to race).

usage: knowledge_share.py CHECKPOINT [--bytes N] [--window W]   (from the repository root)
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402


def shares(model, tokens):
    """Per level: the knowledge cells' share of each head's cell spikes as [heads, bytes of the window]."""
    out = {}

    def hook(index):
        def grab(read, args, _output):
            fired, _, _, _ = read._address(args[0], args[2] if len(args) > 2 else None)
            known = fired[..., -read.knowledge:].sum(-1)
            out[index] = (known / fired.sum(-1).clamp_min(1)).mean(0).double().cpu().numpy()
        return grab

    handles = [read.register_forward_hook(hook(index)) for index, read in model.state_reads.items() if read.knowledge]
    with torch.no_grad():
        model(tokens)
    for handle in handles:
        handle.remove()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--bytes", type=int, default=16 * 256)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval()
    _, validation, _ = load_splits(Path("data/enwik8"))
    rows = args.bytes // args.window
    tokens = torch.from_numpy(np.array(validation[:rows * args.window], dtype=np.int64)).view(rows, args.window)
    found = shares(model, tokens)
    report = {"run": checkpoint.get("run_name", str(args.checkpoint)), "levels": {}}
    for index in sorted(found, key=int):
        share = found[index]
        early, late = share[:, :16].mean(1), share[:, -64:].mean(1)
        report["levels"][index] = {"by_head": share.mean(1).round(4).tolist(),
                                   "first_16_bytes": early.round(4).tolist(), "last_64_bytes": late.round(4).tolist()}
        print(f"level {index}: knowledge share by head {np.round(share.mean(1), 3).tolist()}, "
              f"first 16 bytes {np.round(early, 3).tolist()}, last 64 {np.round(late, 3).tolist()}", flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
