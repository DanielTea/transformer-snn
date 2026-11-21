"""The gain of an arm over a reference, overall and by quarter of every stream."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from eval_stream import stream_rows  # noqa: E402
from stream_quarters import gains, quarters  # noqa: E402


def test_a_gain_that_grows_along_the_streams_is_seen_quarter_by_quarter():
    data = torch.from_numpy(np.random.default_rng(3).integers(0, 256, 40001, dtype=np.uint8))
    _, _, scored = stream_rows(data, 4, 1024)
    quarter, counts = quarters(scored, 512, 1024)
    # The lead is no quarter, nor is the first stream's start, which it scores from an empty state.
    assert (quarter[:, :2] == -1).all() and counts[0, :2].sum() == 1024 and counts[1:, :2].sum() == 0
    assert set(np.unique(quarter[:, 2:][counts[:, 2:] > 0])) == {0, 1, 2, 3}
    reference = counts * 2.0
    step = np.where(quarter >= 0, 0.01 * (quarter + 1), 0.0)
    arm = counts * (2.0 - step)
    row = gains(reference, arm, counts, quarter)
    assert [part["gain"] for part in row["quarters"]] == pytest.approx([0.01, 0.02, 0.03, 0.04])
    # Every piece of a quarter gains alike, so its interval closes on the gain.
    assert row["targets"] == 40000 and row["quarters"][0]["gain_95"] == pytest.approx([0.01, 0.01])
    # The overall gain weighs every quarter by its bytes and counts the first stream's start as no gain.
    assert row["gain"] == pytest.approx(float((reference - arm).sum() / counts.sum()))
