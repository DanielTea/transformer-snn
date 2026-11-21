"""How well does each test-bed model copy? Random bytes shown twice: the first copy cannot be predicted, the second can
only by finding where the same bytes came before and reading what followed (the induction circuit of Transformers).
Prints, for each checkpoint, the bits a byte on the first copy, on the second, and on the second after its first 8
bytes, together with the model's validation score in windows from its log.
usage: copy_probe.py CHECKPOINT [CHECKPOINT...]"""
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, "experiments")
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402
from snnformer.event_lm import EventLM  # noqa: E402

torch.set_num_threads(4)
import re
import numpy as np
from train_lm import load_splits

generator = torch.Generator().manual_seed(7)
half = 100
# Random lowercase letters, and random words of the enwik8 training split joined by spaces: text-like bytes that the
# models have seen, in orders they cannot predict on the first copy.
letters = torch.randint(97, 123, (64, half), generator=generator)
train, _, _ = load_splits(Path("data/enwik8"))
words = sorted(set(re.findall(rb"[a-z]{3,9}", bytes(np.array(train[:2_000_000], dtype=np.uint8)))))
rng = np.random.default_rng(7)
rows = []
for _ in range(64):
    text = b""
    while len(text) < half:
        text += words[rng.integers(len(words))] + b" "
    rows.append(list(text[:half]))
wordy = torch.tensor(rows)
probes = {"letters": letters, "words": wordy}
for path in sys.argv[1:]:
  model, checkpoint = load_checkpoint(Path(path), torch.device("cpu"))
  model.eval()
  for probe, sequences in probes.items():
    tokens = torch.cat([sequences, sequences], 1)
    with torch.inference_mode():
        output = model(tokens[:, :-1])
        event = isinstance(model, EventLM)
        logits = scoring_logits(output, event, model.cfg.resolution if event else 1)
        bits = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none").view(64, -1) / math.log(2)
    # bits[:, i] is the loss of predicting byte i + 1.
    first = bits[:, :half - 1].mean().item()
    second = bits[:, half - 1:].mean().item()
    settled = bits[:, half - 1 + 8:].mean().item()
    print(f"{checkpoint.get('run_name', path)} [{probe}]: first copy {first:.3f}, second copy {second:.3f}, "
          f"second after 8 bytes {settled:.3f} bits a byte", flush=True)
