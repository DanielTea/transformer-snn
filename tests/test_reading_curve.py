"""The gain of fast synapses against how much of a stream they have read."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def reading(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import reading_curve

    return reading_curve


def test_the_bins_double_after_the_first_eight_kilobytes(reading):
    assert reading.bins(5) == [0, 8]
    assert reading.bins(100) == [0, 8, 16, 32, 64, 128]


def test_bytes_are_counted_piece_by_piece_from_the_scored_places(reading):
    scored = torch.zeros(2, 1100, dtype=torch.bool)
    scored[0, :1100] = True
    scored[1, 30:700] = True
    assert reading.piece_bytes(scored, 512).tolist() == [[512, 512, 76], [482, 188, 0]]


def test_the_gain_pools_the_streams_and_keeps_their_own(reading):
    # Two streams of 40 pieces of 512 bytes: 20 KB each. The first gains one bit a piece, the second three.
    frozen = np.full((2, 40), 10.0)
    taught = frozen - np.array([[1.0], [3.0]])
    counts = np.full((2, 40), 512)
    counts[1, 30:] = 0
    first, second, third = reading.curve(frozen, taught, counts, 512, reading.bins(20))
    # The first bin holds the pieces at 0 to 7.5 KB read, the second those at 8 to 15.5 KB, the third the rest.
    assert (first["from_kb"], first["to_kb"], first["bytes"], first["streams"]) == (0, 8, 16 * 512 * 2, 2)
    assert first["gain"] == pytest.approx((16 * 1 + 16 * 3) / (16 * 512 * 2))
    assert first["median"] == pytest.approx(2 / 512)
    # Bits where nothing is scored do not count, nor do their bytes, nor a stream that has none in a bin.
    assert second["bytes"] == (16 + 14) * 512 and second["gain"] == pytest.approx((16 + 14 * 3) / (30 * 512))
    assert (third["bytes"], third["streams"]) == (8 * 512, 1) and third["gain"] == pytest.approx(1 / 512)
