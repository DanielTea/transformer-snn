"""The control to a stack's fast output synapses: a Transformer that every byte of a stream teaches."""

from pathlib import Path

import pytest
import torch

from snnformer.config import SNNConfig
from snnformer.transformer_reference import TransformerLM


@pytest.fixture
def scorers(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_taught_reference
    import eval_warmed_windows

    return eval_taught_reference, eval_warmed_windows


def _transformer(seed=0):
    torch.manual_seed(seed)
    model = TransformerLM(SNNConfig(neuron="ann", attn="transformer", position_embeddings=True),
                          vocab=256, dim=32, depth=2, heads=4, ctx=32)
    return model.eval()


def _bytes(length, seed=1):
    return torch.randint(0, 256, (length,), generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("context, history", [(8, 24), (8, 8), (16, 0), (5, 11)])
def test_without_teaching_one_stream_scores_as_warmed_windows(scorers, context, history):
    taught, warmed = scorers
    model, data = _transformer(), _bytes(203)
    bits, targets = taught.score(model, data, 1, 0, context, history, [0.0], torch.device("cpu"))
    windows, _, counted = warmed.score(model, [data], context, history, 4, torch.device("cpu"))
    assert targets == counted == 202 and bits.shape[:2] == (1, 1)
    assert bits.sum() == pytest.approx(windows.sum(), rel=1e-5)


def test_rates_are_scored_side_by_side_and_windows_only_share_a_pass(scorers):
    taught, _ = scorers
    model, data = _transformer(), _bytes(400)
    together, targets = taught.score(model, data, 2, 20, 8, 24, [0.0, 0.1, 0.4], torch.device("cpu"))
    alone, _ = taught.score(model, data, 2, 20, 8, 24, [0.4], torch.device("cpu"))
    assert together.shape[:2] == (3, 2) and targets == 399
    assert together[2].sum() == pytest.approx(alone.sum(), rel=1e-5)
    assert together[0].sum() != pytest.approx(together[2].sum(), rel=1e-4)
    # Windows that share a pass are the windows that go one at a time.
    single, _ = taught.score(model, data, 2, 20, 8, 24, [0.0, 0.1, 0.4], torch.device("cpu"), batch=1)
    # Every eight bytes of a stream are one entry, however many windows share a pass.
    assert together.shape == single.shape == (3, 2, 27) and together == pytest.approx(single, rel=1e-4, abs=1e-4)


def test_no_byte_is_learned_from_before_it_is_predicted(scorers):
    taught, _ = scorers
    model, data = _transformer(), _bytes(300)
    changed = data.clone()
    changed[161:] = _bytes(300, seed=7)[161:]
    first, _ = taught.score(model, data, 1, 0, 8, 24, [0.3], torch.device("cpu"))
    second, _ = taught.score(model, changed, 1, 0, 8, 24, [0.3], torch.device("cpu"))
    # A window predicts eight bytes. The first twenty predict the bytes up to place 160, which did not change.
    assert first[..., :20] == pytest.approx(second[..., :20], rel=1e-6)
    assert first[..., 20:] != pytest.approx(second[..., 20:], rel=1e-4)


def test_fast_synapses_learn_a_stream_that_repeats(scorers):
    taught, _ = scorers
    model = _transformer()
    data = _bytes(24, seed=3).repeat(40)
    bits, _ = taught.score(model, data, 1, 0, 8, 24, [0.0, 0.5], torch.device("cpu"))
    late = bits[:, 0, bits.shape[-1] // 2:].sum(-1)
    # An untrained Transformer predicts none of it. Its fast synapses learn most of the repeated bytes.
    assert late[1] < 0.6 * late[0]


def test_every_stream_teaches_its_own_synapses(scorers):
    taught, _ = scorers
    model, data = _transformer(), _bytes(500)
    both, _ = taught.score(model, data, 2, 0, 8, 24, [0.3], torch.device("cpu"))
    # With no lead the first of two parts is the first half of the split, scored as one stream by itself.
    alone, _ = taught.score(model, data[:251], 1, 0, 8, 24, [0.3], torch.device("cpu"))
    assert both[0, 0].sum() == pytest.approx(alone.sum(), rel=1e-5)
