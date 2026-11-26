"""Balanced synapses: every neuron of a stack's reads, plain cells and output has synapses that sum to zero over the
stream's channels, so it answers the stream's pattern and not its level."""

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _balanced_ticks, _centred
from test_stack_loop import _stack
from test_teaching_spikes import _STACK, _bytes, _pieces


def test_balanced_ticks_are_whole_and_sum_to_nearly_zero():
    ticks = torch.randint(-50, 80, (6, 32)).float()
    balanced = _balanced_ticks(ticks)
    assert torch.equal(balanced, balanced.round()) and (balanced.sum(-1).abs() <= 32 / 2).all()
    grouped = _balanced_ticks(ticks, 8).unflatten(-1, (4, 8))
    assert (grouped.sum(-1).abs() <= 8 / 2).all()
    assert torch.equal(_balanced_ticks(ticks.int()), balanced.int())
    torch.testing.assert_close(_centred(ticks).sum(-1), torch.zeros(6), atol=1e-4, rtol=0)


def test_off_is_the_stack_as_it_was():
    plain, tokens = _stack(), _bytes()
    assert plain.output.balance is None
    assert all(not layer.balanced for layer in plain.granule_layers.values())


def test_a_balanced_neuron_hears_the_pattern_and_not_the_level():
    model = _stack(balanced_synapses=True)
    read = model.state_reads["0"]
    assert model.output.balance == _STACK["dim"] and read.payload.balance == _STACK["dim"]
    for synapse in (model.output, read.payload):
        ticks = synapse.weight_ticks()
        level = torch.ones(ticks.shape[-1])
        # A level added to every channel moves a neuron's charge by its ticks' sum: within half a tick per synapse.
        assert ((ticks @ level).abs() <= ticks.shape[-1] / 2).all()
    for layer in model.granule_layers.values():
        assert (layer.receive_ticks().sum(-1).abs() <= layer.receive.shape[-1] / 2).all()


def test_a_balanced_stack_streams_the_same_however_the_stream_is_cut():
    model, tokens = _stack(balanced_synapses=True), _bytes()
    with torch.no_grad():
        whole, _ = model.stream(tokens)
    cut, _ = _pieces(model, tokens, (7, 19, tokens.shape[1] - 26))
    assert torch.equal(whole, cut)


def test_balanced_synapses_learn_and_shrink_by_their_balanced_sizes():
    from snnformer.event_lm import _unrounded_sums

    model = _stack(dtype=torch.float32, balanced_synapses=True).train()
    model(_bytes())[..., 0].sum().backward()
    assert model.output.weight.grad.abs().sum() > 0
    read = next(iter(model.state_reads.values()))
    with torch.no_grad():
        read.payload.weight.copy_(torch.randn_like(read.payload.weight))
        limit = 2000.0
        assert read.scale_synapses(limit) > 0
        bits = read.payload.charge_bits
        spare = read.payload.weight.shape[-1]
        assert (_unrounded_sums(read.payload.sizing_weight(), bits) <= limit - spare / 2).all()
        # The rounded, balanced ticks that the neurons use stay within the limit.
        assert (read.payload.weight_ticks().abs().sum(-1) <= limit).all()
    model._check_accumulator()


@pytest.mark.parametrize("settings", [dict(stack=False), dict(granule_gate=4), dict(balanced_synapses=1)])
def test_balanced_synapses_belong_to_a_stack_of_plain_cells(settings):
    with pytest.raises(ValueError, match="balanced synapses"):
        EventLM(EventLMConfig(**dict(_STACK, **{"balanced_synapses": True, **settings})))
