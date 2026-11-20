from pathlib import Path

import pytest
import torch


@pytest.fixture
def evaluator(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_warmed_windows

    return eval_warmed_windows


class _NeedsHistory(torch.nn.Module):
    """Names the next byte of a counting passage once it has read enough bytes, and nothing before."""

    def __init__(self, needed, confidence=1e4):
        super().__init__()
        self.needed, self.confidence = needed, confidence

    def forward(self, tokens):
        sure = torch.arange(1, tokens.shape[1] + 1) >= self.needed
        logits = torch.zeros(*tokens.shape, 256)
        evidence = (self.confidence * sure).view(1, -1, 1).expand(tokens.shape[0], -1, 1)
        return logits.scatter(2, ((tokens + 1) % 256).unsqueeze(2), evidence)


@pytest.mark.parametrize("targets", (4, 8, 9, 16, 17, 23, 41, 64))
@pytest.mark.parametrize("history", (0, 3, 8, 11, 16, 29))
def test_windows_score_every_target_once_behind_its_history(evaluator, targets, history):
    data = torch.arange(targets + 1)
    scored = []
    for windows, skipped in evaluator.window_batches(data, 8, history, 3):
        assert windows.shape[0] <= 3
        for window in windows:
            assert torch.equal(window, torch.arange(int(window[0]), int(window[0]) + window.numel()))
            # Only a window that starts the passage scores targets with less than the history before them.
            assert skipped == history or int(window[0]) == 0
            assert window.numel() - skipped - 1 <= 8 and window.numel() <= history + 8 + 1
            scored.append(window[skipped + 1:])
    assert torch.equal(torch.cat(scored), data[1:])


def test_a_history_longer_than_the_scored_bytes_fills_the_window(evaluator):
    passages = [torch.arange(70) % 256]
    # A model that needs 20 bytes: with 24 bytes of history before every 8 scored ones only the first 19 targets
    # of the passage are read too early, and with 8 bytes of history every window's first 12 scored ones are.
    model = _NeedsHistory(needed=20)
    bits, whole, targets = evaluator.score(model, passages, 8, 24, 4, torch.device("cpu"))
    assert targets == 69 and bits.sum() == pytest.approx(19 * 8, rel=1e-6)
    assert whole.tolist() == [False] * 3 + [True] * 5 + [False]
    short, _, _ = evaluator.score(model, passages, 8, 8, 4, torch.device("cpu"))
    assert short.sum() == pytest.approx(69 * 8, rel=1e-6)


def test_scored_targets_have_the_history_before_them(evaluator):
    passages = [torch.arange(45) % 256, torch.arange(100, 120) % 256]
    model = _NeedsHistory(needed=9)
    bits, whole, targets = evaluator.score(model, passages, 8, 8, 4, torch.device("cpu"))
    assert targets == 44 + 19
    # Only the first eight targets of each passage are read from an empty state.
    assert bits.sum() == pytest.approx(2 * 8 * 8, rel=1e-6)
    assert bits.shape == (1, len(whole))
    assert whole.tolist() == [False, True, True, True, True, False, False, True, False]
    cold, whole, targets = evaluator.score(model, passages, 8, 0, 4, torch.device("cpu"))
    assert cold.sum() == pytest.approx(8 * targets, rel=1e-6)
    assert whole.tolist() == [True] * 5 + [False] + [True] * 2 + [False]


def test_temperatures_are_scored_side_by_side(evaluator):
    passage = torch.arange(33) % 256
    model = _NeedsHistory(needed=1, confidence=4.0)
    bits, _, targets = evaluator.score(model, [passage], 8, 8, 4, torch.device("cpu"), (1.0, 2.0))
    expected = [torch.log2(1 + 255 * torch.exp(torch.tensor(-4.0 / temperature))) for temperature in (1.0, 2.0)]
    assert bits.shape[0] == 2 and targets == 32
    assert bits.sum(1).tolist() == pytest.approx([float(value) * targets for value in expected], rel=1e-5)
