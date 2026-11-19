"""Writing text with a model that reads windows."""

from pathlib import Path

import pytest
import torch


@pytest.fixture
def experiments(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))


class _Echo(torch.nn.Module):
    """Predicts the byte after the first byte of its window, so its output tells how far back it was shown."""

    def forward(self, tokens):
        logits = torch.zeros(*tokens.shape, 256)
        logits.scatter_(2, ((tokens[:, :1] + 1) % 256).expand_as(tokens).unsqueeze(-1), 5.0)
        return logits


def test_each_byte_is_predicted_from_the_last_bytes_only(experiments):
    from generate_windows import continue_bytes

    # With three bytes in view the model sees the window slide: 10 11 12 | 11 12 11 | 12 11 12 | ...
    written = continue_bytes(_Echo(), bytes([10, 11, 12]), 5, context=3)
    assert list(written) == [11, 12, 13, 12, 13]
    # A longer view keeps the first byte in reach.
    assert list(continue_bytes(_Echo(), bytes([10, 11, 12]), 3, context=8)) == [11, 11, 11]


def test_drawn_bytes_follow_the_seed_and_the_temperature(experiments):
    from generate_windows import continue_bytes

    def draw(seed, temperature):
        return continue_bytes(_Echo(), b"abc", 40, context=3, temperature=temperature,
                              generator=torch.Generator().manual_seed(seed))

    assert draw(1, 4.0) == draw(1, 4.0) != draw(2, 4.0)
    # A cold draw is the most likely byte almost always.
    assert sum(a == b for a, b in zip(draw(1, 0.05), continue_bytes(_Echo(), b"abc", 40, context=3))) >= 39
