"""An energy pool: a stack's inhibitory pools hear the stream's energy and so divide it by its spread."""

import torch

import pytest

from snnformer.event_lm import EventLM, EventLMConfig, _heard_activity
from test_stack_loop import _stack
from test_teaching_spikes import _STACK, _bytes, _pieces


def test_the_energy_of_a_stream_whose_channels_fire_alike_is_its_summed_counts():
    alike = torch.zeros(2, 3, 64)
    alike[..., :16] = 3.0
    # Sixteen channels at three: summed counts 48; energy sqrt(64 * 16 * 9) = 96, the summed counts over all 64
    # channels had they all fired alike at the mean square.
    assert torch.equal(_heard_activity(alike), torch.full((2, 3, 1), 48.0))
    every = torch.full((1, 1, 64), 2.0)
    assert torch.equal(_heard_activity(every, True), _heard_activity(every))


def test_a_few_loud_channels_weigh_more_in_the_energy():
    quiet, loud = torch.ones(1, 1, 64), torch.zeros(1, 1, 64)
    loud[..., 0] = 64.0
    assert torch.equal(_heard_activity(quiet), _heard_activity(loud))
    assert _heard_activity(loud, True) > 7 * _heard_activity(quiet, True)
    # Whole counts in, whole counts out; the gradient is that of the smooth root.
    counts = torch.randint(0, 9, (2, 5, 32)).float().requires_grad_()
    heard = _heard_activity(counts, True)
    assert torch.equal(heard.detach(), heard.detach().floor())
    heard.sum().backward()
    assert counts.grad.abs().sum() > 0


def test_an_energy_pool_streams_the_same_however_the_stream_is_cut():
    model, tokens = _stack(pool_energy=True), _bytes()
    assert all(read.energy for read in model.state_reads.values())
    with torch.no_grad():
        whole, _ = model.stream(tokens)
        plain = _stack()(tokens)
    cut, _ = _pieces(model, tokens, (11, 23, tokens.shape[1] - 34))
    assert torch.equal(whole, cut) and not torch.equal(whole, plain)


def test_an_energy_pool_belongs_to_a_stack():
    with pytest.raises(ValueError, match="energy pool"):
        EventLM(EventLMConfig(**dict(_STACK, stack=False, granule_after=None, read_after=None, pool_energy=True)))
