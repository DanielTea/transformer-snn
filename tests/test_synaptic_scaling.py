import copy
from pathlib import Path

import pytest
import torch

from snnformer.event_lm import _FP32_EXACT, _PHASE_SPIKES, EventLM, EventLMConfig, _shrink_synapses, _unrounded_sums

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True)


@pytest.fixture
def experiments(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # Reads and cell layers start silent. Give them output synapses.
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _limit(model):
    return _FP32_EXACT // model.input_emit.cap


def _tighten(model):
    """A 64-wide model is far inside the range of exact sums. Let its stream neurons count so far that the range
    ends between the sums of the neurons as they start and the sums of neurons with synapses at their largest."""
    model.input_emit.cap = 4096
    assert model.synaptic_scaling() == 0
    return model


def _families(model):
    """Every weight whose neurons the scaling watches, by name."""
    weights = {"output": model.output.weight}
    for level, read in model.state_reads.items():
        for name in ("probe", "tag", "payload", "project"):
            weights[f"read {level} {name}"] = getattr(read, name).weight
    for level, layer in model.granule_layers.items():
        weights[f"cells {level} receive"], weights[f"cells {level} send"] = layer.receive, layer.send
        if layer.gated:
            weights[f"cells {level} gate"] = layer.gate
    return weights


def test_a_model_within_the_range_of_exact_sums_is_left_as_it_is():
    model = _stack()
    before = copy.deepcopy(model.state_dict())
    assert model.synaptic_scaling() == 0
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())


def test_the_neurons_beyond_the_range_shrink_and_no_other_synapse_moves():
    model = _tighten(_stack())
    grown = {}
    with torch.no_grad():
        for name, weight in _families(model).items():
            neuron = (0,) * (weight.ndim - 1)
            weight[neuron] = 1.9 * torch.sign(weight[neuron] + 1e-9)
            grown[name] = neuron
    with pytest.raises(ValueError, match="exact FP32 accumulator range"):
        model._check_accumulator()
    before = {name: weight.detach().clone() for name, weight in _families(model).items()}
    scaled = model.synaptic_scaling()
    assert scaled >= len(grown)
    model._check_accumulator()
    for name, weight in _families(model).items():
        neuron = grown[name]
        factor = weight[neuron] / before[name][neuron]
        # One factor below one for all synapses of a shrunk neuron: their proportions stay.
        assert factor.max() < 1 and torch.allclose(factor, factor[0].expand_as(factor), rtol=1e-5), name
        if "send" in name:
            continue
        untouched = torch.ones(weight.shape[:-1], dtype=torch.bool)
        untouched[neuron] = False
        assert torch.equal(weight[untouched], before[name][untouched]), name
    # A second pass finds nothing left to do.
    assert model.synaptic_scaling() == 0


def test_a_stream_channel_shrinks_all_the_synapses_that_reach_it():
    model = _tighten(_stack(granule_clusters=2, granule_fan=32, granule_cells=48))
    layer = model.granule_layers["0"]
    channel = int(layer.outputs[0, 0])
    with torch.no_grad():
        layer.send[layer.outputs == channel] = 1.9
    before = layer.send.detach().clone()
    assert model.synaptic_scaling() >= 1
    model._check_accumulator()
    reaching = layer.outputs == channel
    factor = layer.send[reaching] / before[reaching]
    assert factor.max() < 1 and torch.allclose(factor, factor.flatten()[0].expand_as(factor), rtol=1e-5)
    assert torch.equal(layer.send[~reaching], before[~reaching])


def test_gated_cells_and_reads_with_pacemakers_stay_in_range_too():
    model = _tighten(_stack(granule_gate=4, granule_cells=64, read_phases=4))
    with torch.no_grad():
        for layer in model.granule_layers.values():
            layer.gate.fill_(1.9)
            layer.receive.fill_(-1.9)
        for read in model.state_reads.values():
            read.probe.weight.fill_(1.9)
            read.tag.weight.fill_(1.9)
            read.probe_phase.weight.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator range"):
        model._check_accumulator()
    assert model.synaptic_scaling() > 0
    model._check_accumulator()
    read = model.state_reads["0"]
    heard = (read.probe_phase.weight_ticks().abs().sum(-1) * (_PHASE_SPIKES / read.emit.cap)).ceil()
    assert (read.probe.weight_ticks().abs().sum(-1) + heard).max() <= _limit(model)


def test_a_heads_grain_counts_towards_what_a_read_neuron_sums():
    model = _tighten(_stack())
    read = model.state_reads["1"]
    with torch.no_grad():
        read.project.weight.fill_(1.5)
        read.grain.fill_(4.0)
    with pytest.raises(ValueError, match="exact FP32 accumulator range"):
        model._check_accumulator()
    assert model.synaptic_scaling() == read.project.weight.shape[0]
    model._check_accumulator()
    assert read.project.weight.max() < 1.5


def test_shrinking_treats_a_synapse_beyond_the_largest_weight_as_the_largest():
    weight = torch.tensor([[3.0, -0.5, 0.25], [0.1, 0.2, 0.3]])
    sums = _unrounded_sums(weight, 8)
    assert sums.tolist() == [508 + 128 + 64, pytest.approx(25.6 + 51.2 + 76.8)]
    assert _shrink_synapses(weight, torch.tensor([0.5, 2.0]), 8) == 1
    assert weight[0].tolist() == pytest.approx([508 / 256 / 2, -0.25, 0.125])
    assert weight[1].tolist() == pytest.approx([0.1, 0.2, 0.3])


def test_a_decoder_with_integer_weights_has_nothing_to_scale():
    model = _stack()
    model.input.freeze_integer()
    assert model.synaptic_scaling() == 0


def test_the_average_of_the_weights_is_scored_within_the_range(experiments):
    from train_event_lm import averaged_weights

    model = _tighten(_stack())
    own = copy.deepcopy(model.state_dict())
    # An average that pairs the large read synapses of one time with the large grain of another.
    average = [parameter.detach().clone() for parameter in model.parameters()]
    for (name, _), mean in zip(model.named_parameters(), average, strict=True):
        if name == "state_reads.1.project.weight":
            mean.fill_(1.5)
        elif name == "state_reads.1.grain":
            mean.fill_(4.0)
    lent = [mean.clone() for mean in average]
    with averaged_weights(model, average):
        model._check_accumulator()
        assert model.state_reads["1"].project.weight.max() < 1.5
        assert torch.equal(model.state_reads["1"].grain, torch.full_like(model.state_reads["1"].grain, 4.0))
    # The model has its own weights again, and the average itself is as it was.
    assert all(torch.equal(value, own[name]) for name, value in model.state_dict().items())
    assert all(torch.equal(mean, before) for mean, before in zip(average, lent, strict=True))
