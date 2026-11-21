"""Cells whose gate is a shunting conductance, and an output that hears the stream through the input's synapses."""

from dataclasses import asdict

import pytest
import torch

from snnformer.event_lm import _PACKET_UNITS, EventLM, EventLMConfig, _GatedFire
from test_teaching_spikes import _STACK, _bytes, _pieces

_SHUNT = dict(granule_gate=4, granule_shunt=True)
_SELF = dict(granule_gate=4, granule_self_gate=True, granule_shunt=True)


def _stack(seed=0, dtype=torch.float32, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model.to(dtype).eval()


def _bodies(unit=4, quanta_per_threshold=_PACKET_UNITS):
    plain = _GatedFire(64, 6, unit, quanta_per_threshold)
    shunted = _GatedFire(64, 6, unit, quanta_per_threshold, shunt=True)
    return plain, shunted


def test_a_shunting_gate_never_lets_in_more_than_all_of_the_charge():
    plain, shunted = _bodies()
    quanta = torch.ones(1)
    whole = plain.threshold // plain.quanta_per_threshold
    generator = torch.Generator().manual_seed(3)
    charge = torch.randint(-40 * whole, 40 * whole, (500,), generator=generator).double()
    gate = torch.randint(0, 64, (500,), generator=generator).double()
    counts = shunted(charge, quanta, opened=gate)
    # What a plain cell fires for the collected charge: a gate of `unit` spikes under a multiplying gate.
    alone = plain(charge, quanta, opened=torch.full_like(gate, plain.unit))
    assert (counts.abs() <= alone.abs()).all() and (counts * alone >= 0).all()
    # At one threshold of the gate's charge it lets in half, and with a closed gate nothing.
    half = shunted(charge, quanta, opened=torch.full_like(gate, shunted.unit))
    assert ((half - plain(charge / 2, quanta, opened=torch.full_like(gate, plain.unit))).abs() <= 1).all()
    assert not shunted(charge, quanta, opened=torch.zeros_like(gate)).any()
    # A multiplying gate lets in up to sixteen times as much.
    assert (plain(charge, quanta, opened=gate).abs() > alone.abs()).any()


def test_a_cell_that_gates_itself_through_a_shunt_fires_a_square_then_a_line():
    _, shunted = _bodies()
    whole = shunted.threshold // shunted.quanta_per_threshold
    thresholds = torch.arange(0, 25).double()
    counts = shunted(thresholds * whole, torch.ones(1), opened=thresholds * shunted.unit)
    expected = torch.floor(thresholds * thresholds / (thresholds + 1) + 0.5)
    assert torch.equal(counts, expected)
    assert (counts.diff() >= 0).all() and counts[1] == 1 and counts[-1] == 23


def test_the_shunt_passes_its_slope_to_the_value_and_the_gate():
    _, shunted = _bodies()
    whole = shunted.threshold // shunted.quanta_per_threshold
    charge = torch.tensor([3.0 * whole], requires_grad=True)
    gate = torch.tensor([8.0], requires_grad=True)
    shunted(charge, torch.ones(1), opened=gate).sum().backward()
    # d/dv of v k / (k + u) is k / (k + u) and d/dk is v u / (k + u) ** 2, in thresholds.
    assert torch.allclose(charge.grad * whole, torch.tensor([8.0 / 12.0]))
    assert torch.allclose(gate.grad, torch.tensor([3.0 * 4.0 / 144.0]))


@pytest.mark.parametrize("design", [_SHUNT, _SELF, dict(output_tied=True), dict(_SHUNT, output_tied=True)])
def test_a_stream_fires_what_its_window_fires_however_it_is_cut(design):
    model, tokens = _stack(dtype=torch.float64, **design), _bytes(length=200)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (200,))
    assert torch.equal(window, whole)
    for cuts in ((1,) * 200, (7,) * 28 + (4,), (128, 72)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)


def test_a_shunted_cell_trains_its_gate_and_its_value():
    model = _stack(**_SHUNT).train()
    model(_bytes()).float().square().mean().backward()
    for layer in model.granule_layers.values():
        assert layer.gate.grad.abs().sum() > 0 and layer.receive.grad.abs().sum() > 0


def test_the_output_hears_through_the_input_synapses():
    plain, tied = _stack(), _stack(output_tied=True)
    assert tied.output.weight is tied.input.weight
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    assert count(plain) - count(tied) == 256 * _STACK["dim"]
    # The threshold of an output neuron rises by the power of two nearest 0.9 sqrt(dim) / 0.8: 2 ** 3 at 64.
    assert tied.output_emit.threshold == plain.output_emit.threshold * 8
    # Training keeps one row per byte: the scaling of synapses after a step shrinks the shared row in place.
    tied.synaptic_scaling()
    assert tied.output.weight is tied.input.weight
    assert [name for name, _ in tied.named_parameters()].count("input.weight") == 1
    rebuilt = EventLM(EventLMConfig(**asdict(tied.cfg)))
    rebuilt.load_state_dict(tied.state_dict())
    assert rebuilt.output.weight is rebuilt.input.weight and torch.equal(rebuilt.input.weight, tied.input.weight)


@pytest.mark.parametrize("setting", [dict(granule_shunt=True), dict(granule_shunt=True, granule_after=None),
                                     dict(granule_shunt=1, granule_gate=4)])
def test_a_shunt_is_the_gate_of_a_gated_cell(setting):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **setting)))


def test_only_a_stack_shares_its_input_synapses_with_its_output():
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(output_tied=True))


def test_the_trainer_trains_shunted_cells_and_a_tied_output(monkeypatch, tmp_path):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / "run", "--granule-gate", "4", "--granule-shunt",
                    "--tied-output")
    assert record["cfg"]["granule_shunt"] and record["cfg"]["output_tied"]
    saved = torch.load(tmp_path / "run" / "model.pt", weights_only=True)
    rebuilt = EventLM(EventLMConfig(**saved["cfg"]))
    rebuilt.load_state_dict(saved["state_dict"])
    assert rebuilt.output.weight is rebuilt.input.weight
