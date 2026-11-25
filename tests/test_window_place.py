"""The bits of a stack by a byte's place in its sleep window add up to the stream scorer's bits."""

from pathlib import Path
import sys

import numpy as np
import torch

import test_lifted_ceiling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from eval_stream import score, stream_rows  # noqa: E402
from window_place import bits_by_place  # noqa: E402


def test_places_add_up_to_the_stream_score_and_count_every_scored_byte():
    model = test_lifted_ceiling._stack(0, stack_sleep=16).double().eval()
    data = torch.from_numpy(np.random.default_rng(4).integers(0, 256, 700, dtype=np.uint8))
    rows = stream_rows(data, 3, 64)
    bits, targets = score(model, rows, 50, torch.device("cpu"))
    sums, counts = bits_by_place(model, rows, 50, torch.device("cpu"), 16)
    assert int(counts.sum()) == targets
    assert np.isclose(float(sums.sum()), float(bits.sum()), rtol=1e-12)
    # Every place of the window is scored about equally often.
    assert counts.min() > 0 and counts.max() - counts.min() <= rows[2].shape[0] * 4
