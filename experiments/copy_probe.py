"""How well does each model copy? Random letters, and random words of the enwik8 training split, each shown twice in
one window: the first copy cannot be predicted beyond the statistics of letters and words, the second can by finding
the same bytes earlier and reading what followed them (the induction circuit of Transformers). Prints, for each
checkpoint and probe, the bits a byte on the first copy, on the second, and on the second after its first 8 bytes.

usage: copy_probe.py CHECKPOINT [CHECKPOINT...]   (from the repository root)
"""
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from train_lm import load_splits  # noqa: E402
from snnformer.event_lm import EventLM  # noqa: E402

HALF = 100


def probes():
    """64 random sequences of HALF letters, and 64 of random words joined by spaces."""
    generator = torch.Generator().manual_seed(7)
    letters = torch.randint(97, 123, (64, HALF), generator=generator)
    train, _, _ = load_splits(Path("data/enwik8"))
    words = sorted(set(re.findall(rb"[a-z]{3,9}", bytes(np.array(train[:2_000_000], dtype=np.uint8)))))
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(64):
        text = b""
        while len(text) < HALF:
            text += words[rng.integers(len(words))] + b" "
        rows.append(list(text[:HALF]))
    return {"letters": letters, "words": torch.tensor(rows)}


def main():
    torch.set_num_threads(4)
    shown = probes()
    for path in sys.argv[1:]:
        model, checkpoint = load_checkpoint(Path(path), torch.device("cpu"))
        model.eval()
        event = isinstance(model, EventLM)
        for probe, sequences in shown.items():
            tokens = torch.cat([sequences, sequences], 1)
            with torch.inference_mode():
                logits = scoring_logits(model(tokens[:, :-1]), event, model.cfg.resolution if event else 1)
                bits = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none")
            # Column i is the loss of predicting byte i + 1.
            bits = bits.view(64, -1) / math.log(2)
            first, second = bits[:, :HALF - 1].mean().item(), bits[:, HALF - 1:].mean().item()
            settled = bits[:, HALF - 1 + 8:].mean().item()
            print(f"{checkpoint.get('run_name', path)} [{probe}]: first copy {first:.3f}, second copy {second:.3f}, "
                  f"second after 8 bytes {settled:.3f} bits a byte", flush=True)


if __name__ == "__main__":
    main()
