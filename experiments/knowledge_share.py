"""How much of every state read goes to its knowledge cells, read as a stream: the share of each read's cell spikes
that the knowledge cells take, by level and head, over unbroken streams of the enwik8 validation split taken piece by
piece with the state the stack carries, and how the share moves as the stream goes on (at its first bytes a read has
few cells of the stream to race).

usage: knowledge_share.py CHECKPOINT [--streams N] [--length BYTES] [--piece BYTES] [--json PATH]
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

# Stretches of a stream over which the share is reported, in bytes from its start.
STRETCHES = ((0, 64), (64, 512), (512, 2048), (2048, None))


def record_shares(model):
    """Wrap every read's race so that each call records the knowledge cells' share of each head's spikes, as
    [heads, bytes], and return the list the records go to."""
    found = {}
    for index, read in model.state_reads.items():
        if not read.knowledge:
            continue
        original = read._address

        def recording(*args, _read=read, _original=original, _index=index, **kwargs):
            fired, total, payload, spikes = _original(*args, **kwargs)
            known = fired[..., -_read.knowledge:].sum(-1)
            found.setdefault(_index, []).append((known / fired.sum(-1).clamp_min(1)).mean(0).double().cpu())
            return fired, total, payload, spikes

        read._address = recording
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--piece", type=int, default=256)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    model.eval()
    _, validation, _ = load_splits(Path("data/enwik8"))
    stride = (len(validation) - args.length - 1) // args.streams
    rows = np.arange(args.streams)[:, None] * stride + np.arange(args.length)
    tokens = torch.from_numpy(np.array(validation, dtype=np.uint8)[rows].astype(np.int64))
    found = record_shares(model)
    state = None
    with torch.no_grad():
        for start in range(0, args.length, args.piece):
            _, state = model.stream(tokens[:, start:start + args.piece], state)
    report = {"run": checkpoint.get("run_name", str(args.checkpoint)), "streams": args.streams, "bytes": args.length,
              "levels": {}}
    for index in sorted(found, key=int):
        share = torch.cat(found[index], -1).numpy()
        stretches = {}
        for begin, end in STRETCHES:
            part = share[:, begin:end]
            if part.shape[1]:
                stretches[f"{begin}-{'' if end is None else end}"] = part.mean(1).round(4).tolist()
        report["levels"][index] = {"by_head": share.mean(1).round(4).tolist(), "by_stretch": stretches}
        late = stretches.get(f"{STRETCHES[-1][0]}-", stretches[list(stretches)[-1]])
        print(f"level {index}: share by head {np.round(share.mean(1), 3).tolist()}; first 64 bytes "
              f"{np.round(share[:, :64].mean(1), 3).tolist()}; beyond {STRETCHES[-1][0]} {np.round(late, 3).tolist()}",
              flush=True)
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
