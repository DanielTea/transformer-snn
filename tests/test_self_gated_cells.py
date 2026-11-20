"""A cell that gates itself: a gated cell whose gate dendrite hears the fibres through the cell's own synapses, so it
fires about the square of its charge above the rest, with no synapses beyond a plain cell's."""

import pytest
import torch

from snnformer.event_lm import _GATE_CAP, EventLM, EventLMConfig, _ExcitatoryFire
from test_stream_form import _DELAYED, _LATE, _POOLED, _STACK, _bytes, _pieces

_SQUARE = dict(granule_gate=4, granule_self_gate=True)


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _stream(model, tokens):
    return model.input_emit(model.input(tokens) * model.cfg.resolution)


def test_a_cell_that_gates_itself_has_the_synapses_of_a_plain_cell():
    plain, square = _stack(), _stack(**_SQUARE)
    assert set(square.state_dict()) == set(plain.state_dict())
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    assert count(square) == count(plain)
    assert all(layer.gate is None and layer.gated == 4 for layer in square.granule_layers.values())


@pytest.mark.parametrize("design", [dict(), _POOLED, dict(_LATE, **_DELAYED, **_POOLED)])
def test_it_is_a_gated_cell_whose_gate_hears_through_the_cells_own_synapses(design):
    gated, square = _stack(granule_gate=4, **design), _stack(**_SQUARE, **design)
    square.load_state_dict({key: value for key, value in gated.state_dict().items() if not key.endswith(".gate")})
    with torch.no_grad():
        for layer in gated.granule_layers.values():
            layer.gate.copy_(layer.receive)
    tokens = _bytes()
    assert torch.equal(gated(tokens), square(tokens))


def test_its_count_is_the_square_of_its_charge_in_thresholds():
    layer = _stack(**_SQUARE, **_POOLED).granule_layers["0"]
    # Under a fine pool sixty-four quanta are a whole threshold of 256 ticks; a gate spike is a quarter of one.
    quanta = torch.full((1,), 64.0)
    threshold = 64 * (layer.fire.threshold // layer.fire.quanta_per_threshold)
    charge = torch.arange(-2, 9, dtype=torch.float32) * threshold
    opened = layer.open(charge, quanta)
    assert opened.tolist() == [0, 0, 0, 4, 8, 12, 16, 20, 24, 28, 32]
    # k thresholds of charge open the gate k times over, and each opening lets the k thresholds in once more.
    assert layer.fire(charge, quanta, opened=opened).tolist() == [0, 0, 0, 1, 4, 9, 16, 25, 36, 49, 64]
    # A plain cell answers the same charge with k spikes.
    plain = _ExcitatoryFire(layer.fire.cap, layer.bits, False, quanta_per_threshold=layer.fire.quanta_per_threshold,
                            capped=True)
    assert plain(charge, quanta).tolist() == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    # The rest holds the gate back by its thresholds: three thresholds above a rest of one open it twice over.
    rest = torch.full_like(charge, float(threshold))
    assert layer.fire(charge, quanta, opened=layer.open(charge - rest, quanta))[5].item() == 6
    # The gate counts sixteen thresholds at most, and the cell fires its largest count at most.
    large = torch.tensor([20.0, 40.0]) * threshold
    assert layer.open(large, quanta).tolist() == [_GATE_CAP * 4] * 2
    assert layer.fire(large, quanta, opened=layer.open(large, quanta)).tolist() == [min(320, layer.fire.cap)] * 2


def test_a_stack_of_such_cells_fires_whole_spikes_on_one_rail_and_sees_no_later_byte():
    model, tokens = _stack(**_SQUARE, **_POOLED), _bytes()
    changed = tokens.clone()
    changed[:, 100:] = _bytes(seed=7)[:, 100:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :100], second[:, :100]) and not torch.equal(first[:, 100:], second[:, 100:])
    cells = model.granule_layers["0"].cells(_stream(model, tokens))
    # The gate opens above the rest only, and then lets a positive charge in.
    assert torch.equal(cells, cells.round()) and cells.min() == 0 and (cells > 0).any()


def test_every_synapse_of_such_a_stack_learns():
    model = _stack(**_SQUARE, **_POOLED)
    model(_bytes()).float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


@pytest.mark.parametrize("cuts", [(160,), (47, 1, 49, 63), (7, 13, 100, 40)])
def test_a_stream_of_such_cells_cut_anywhere_fires_the_spikes_of_one_pass(cuts):
    model, tokens = _stack(**_SQUARE, **_LATE, **_DELAYED, **_POOLED), _bytes()
    whole = model(tokens)
    fired, state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole) and state.position == 160


def test_cells_that_gate_themselves_stay_within_the_range_of_exact_sums():
    model = _stack(**_SQUARE, **_POOLED)
    model._check_accumulator()
    layer = model.granule_layers["0"]
    with torch.no_grad():
        layer.receive.fill_(1.9)
    # The cell's synapses are its gate's as well, so their sum carries the gate's resting charge.
    rest = -(-16 * 4 * (4 + layer.pool.cap) * layer.quarter() // layer.emit.cap)
    assert layer.accumulator_bounds()[0] == 64 * 486 + rest
    with torch.no_grad():
        layer.receive.fill_(1000.0)
    assert model.synaptic_scaling() > 0
    model._check_accumulator()


@pytest.mark.parametrize("setting", [dict(granule_self_gate=True), dict(granule_gate=4, granule_self_gate=1)])
def test_a_layer_refuses_to_gate_cells_by_themselves_without_a_gate(setting):
    with pytest.raises(ValueError, match="gates itself"):
        _stack(**setting)


def test_a_stack_without_cell_layers_has_nothing_to_gate():
    with pytest.raises(ValueError, match="granule_self_gate"):
        EventLM(EventLMConfig(**dict(_STACK, granule_after=None, granule_self_gate=True)))


def test_the_gate_of_a_cell_that_gates_itself_is_a_dendrite_that_sends_no_pulse():
    from snnformer.event_lm import _IntegrateFire, emitted_pulses

    model, tokens = _stack(**_SQUARE, **_POOLED), _bytes()
    counted = {}
    for name, module in model.named_modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(
                lambda population, _, counts, name=name: counted.__setitem__(name, emitted_pulses(population, counts)))
    model(tokens)
    gates = {name: pulses for name, pulses in counted.items() if name.endswith(".open")}
    bodies = {name: pulses for name, pulses in counted.items() if name.endswith(".fire") and "granule" in name}
    assert len(gates) == len(bodies) == 2
    # The gates step and the bodies fire, but only the bodies' spikes are pulses.
    assert all(pulses == 0 for pulses in gates.values()) and all(pulses > 0 for pulses in bodies.values())
    # A gated cell with a gate of its own still counts its gate's spikes.
    gated = _stack(granule_gate=4, **_POOLED)
    assert not any(layer.open.dendrite for layer in gated.granule_layers.values())
