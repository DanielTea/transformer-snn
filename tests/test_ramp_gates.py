"""Gated cells whose gate opens in proportion to its count, and gates whose steps stay inside the cell."""

import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import _PACKET_UNITS, EventLM, EventLMConfig, _GatedFire, _IntegrateFire, emitted_pulses
from test_teaching_spikes import _bytes, _pieces

# A gate that fires four spikes for a threshold of its charge and is wide open at eight thresholds: 32 spikes.
_RAMP = dict(granule_gate=4, granule_ramp=8)
_INNER = dict(_RAMP, granule_inner_gate=True)
_SELF = dict(_RAMP, granule_self_gate=True)
# One spike of a body under one quantum of threshold, in ticks of charge.
_SPIKE = 16


def _body(ramp=16, lifted=False, shunt=False):
    # Sixty-four ticks to a whole threshold of four quanta, a membrane of 64 whole thresholds, 64 spikes at most.
    body = _GatedFire(64, 6, 4, _PACKET_UNITS, shunt=shunt, ramp=ramp)
    body.lifted = lifted
    return body


def _stack(lifted, seed=0, **design):
    settings = dict(design, pool_lifts_ceiling=lifted)
    return test_lifted_ceiling._stack(seed, **settings).double().eval()


def _charges(count=500, most=40, seed=3):
    generator = torch.Generator().manual_seed(seed)
    charge = torch.randint(-most * _SPIKE, most * _SPIKE, (count,), generator=generator).double()
    return charge, torch.randint(0, 65, (count,), generator=generator).double()


def test_a_ramp_gate_lets_in_as_many_parts_as_it_fires_spikes():
    ramp, plain, quanta = _body(), _body(ramp=0), torch.ones(1)
    charge, gate = _charges()
    counts = ramp(charge, quanta, opened=gate)
    # Sixteen thresholds of the gate's charge are 64 spikes: k of them let in k parts in 64.
    assert torch.equal(counts, torch.floor(charge.abs() * gate / (_SPIKE * 64) + 0.5) * charge.sign())
    # What a plain cell fires for the collected charge: four spikes of a multiplying gate.
    alone = plain(charge, quanta, opened=torch.full_like(gate, 4))
    assert (counts.abs() <= alone.abs()).all() and (counts * alone >= 0).all()
    # At its largest count the gate lets in all of the charge, and closed it lets in none.
    assert torch.equal(ramp(charge, quanta, opened=torch.full_like(gate, 64)), alone)
    assert not ramp(charge, quanta, opened=torch.zeros_like(gate)).any()
    # A multiplying gate lets in up to sixteen times as much.
    assert (plain(charge, quanta, opened=gate).abs() > alone.abs()).any()


def test_every_gate_spike_lets_in_as_much_as_the_one_before():
    quanta, gate = torch.ones(1), torch.arange(0, 65).double()
    # A value of 64 spikes: every gate spike of the ramp lets in one of them.
    charge = torch.full_like(gate, 64 * _SPIKE)
    assert torch.equal(_body()(charge, quanta, opened=gate), gate)
    # A shunting gate lets in less and less: 43 spikes for its first eight, none for its last eight.
    steps = _body(ramp=0, shunt=True)(charge, quanta, opened=gate).diff()
    assert steps[0] == 13 and steps[:8].sum() == 43 and steps[-8:].sum() == 0


@pytest.mark.parametrize("spikes", [8.0, 40.0])
def test_the_slope_of_a_ramp_does_not_fade_as_the_gate_opens(spikes):
    charge = torch.tensor([3.0 * _SPIKE], requires_grad=True)
    gate = torch.tensor([spikes], requires_grad=True)
    _body()(charge, torch.ones(1), opened=gate).sum().backward()
    # In spikes, d/dv of v k / 64 is k / 64 and d/dk is v / 64 at every k.
    assert torch.allclose(charge.grad * _SPIKE, torch.tensor([spikes / 64]))
    assert torch.allclose(gate.grad, torch.tensor([3.0 / 64]))


def test_a_cell_that_gates_itself_through_a_ramp_fires_a_square_up_to_the_ramps_width_and_a_line_above():
    thresholds = torch.arange(0, 25).double()
    gate = (thresholds * 4).clamp(max=64)
    counts = _body()(thresholds * _SPIKE, torch.ones(1), opened=gate)
    assert torch.equal(counts, torch.floor(thresholds * thresholds.clamp(max=16) / 16 + 0.5))
    assert (counts.diff() >= 0).all() and counts[2] == 0 and counts[3] == 1 and counts[16] == 16 and counts[24] == 24
    # A narrower ramp reaches the line sooner.
    narrow = _body(ramp=4)(thresholds * _SPIKE, torch.ones(1), opened=(thresholds * 4).clamp(max=16))
    assert torch.equal(narrow, torch.floor(thresholds * thresholds.clamp(max=4) / 4 + 0.5))


@pytest.mark.parametrize("lifted", [False, True])
def test_a_ramp_gate_is_wide_open_at_the_most_spikes_it_fires_at_the_present_threshold(lifted):
    ramp, plain = _body(lifted=lifted), _body(ramp=0, lifted=lifted)
    # Eight quanta double the threshold. A bounded gate's membrane then holds 32 of its spikes, a lifted gate's 64.
    quanta, most = torch.tensor([8.0]), 64 if lifted else 32
    charge, _ = _charges(most=100)
    alone = plain(charge, quanta, opened=torch.full_like(charge, 4))
    assert torch.equal(ramp(charge, quanta, opened=torch.full_like(charge, most)), alone)
    half = ramp(charge, quanta, opened=torch.full_like(charge, most // 2))
    assert ((half - plain(charge / 2, quanta, opened=torch.full_like(charge, 4))).abs() <= 1).all()
    assert torch.equal(half, half.round()) and half.abs().max() <= 64
    # Whole numbers of charge fire the same counts.
    gate = torch.full_like(charge, most // 2).to(torch.int32)
    assert torch.equal(ramp(charge.to(torch.int32), quanta.to(torch.int32), opened=gate), half.to(torch.int32))


@pytest.mark.parametrize("lifted", [False, True])
def test_a_ramp_learns_through_its_value_and_its_gate(lifted):
    charge = torch.tensor([10.0 * _SPIKE], requires_grad=True)
    gate = torch.tensor([24.0], requires_grad=True)
    _body(lifted=lifted)(charge, torch.tensor([8.0]), opened=gate).sum().backward()
    assert charge.grad > 0 and gate.grad > 0


@pytest.mark.parametrize("design", [_RAMP, _INNER, _SELF, dict(granule_gate=4, granule_ramp=16),
                                    dict(granule_gate=4, granule_shunt=True, granule_inner_gate=True)])
@pytest.mark.parametrize("lifted", [False, True])
def test_a_stream_fires_what_its_window_fires_however_it_is_cut(design, lifted):
    model, tokens = _stack(lifted, **design), _bytes(length=120)
    fired = []
    handles = [layer.fire.register_forward_hook(lambda _module, _inputs, counts: fired.append(counts.detach()))
               for layer in model.granule_layers.values()]
    with torch.no_grad():
        window = model(tokens)
    for handle in handles:
        handle.remove()
    # The cuts are tested on cells that speak. A body fires on the rail of its value's sign. A cell that gates itself
    # opens only above its rest, so it fires on one rail.
    assert all((counts > 0).any() for counts in fired)
    assert all((counts < 0).any() != design.get("granule_self_gate", False) for counts in fired)
    whole, _ = _pieces(model, tokens, (120,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 120, (7,) * 17 + (1,), (64, 56)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_a_ramp_gate_counts_the_thresholds_at_which_it_is_wide_open():
    for layer in _stack(True, **_RAMP).granule_layers.values():
        assert layer.fire.ramp == layer.open.cap == 8 and layer.open.capped == 32
    # Any other gate counts sixteen thresholds.
    for layer in _stack(True, granule_gate=4, granule_shunt=True).granule_layers.values():
        assert layer.fire.ramp == 0 and layer.open.cap == 16 and layer.open.capped == 64


def _pulses(model, tokens):
    counted, handles = {}, []
    for name, module in model.named_modules():
        if isinstance(module, _IntegrateFire):
            handles.append(module.register_forward_hook(
                lambda population, _, counts, name=name: counted.__setitem__(name, emitted_pulses(population, counts))))
    with torch.no_grad():
        counts = model(tokens)
    for handle in handles:
        handle.remove()
    return counts, counted


@pytest.mark.parametrize("law", [dict(granule_ramp=8), dict(granule_shunt=True)])
def test_the_steps_of_an_inner_gate_stay_inside_the_cell(law):
    tokens = _bytes(length=120)
    firing, inner = _stack(True, granule_gate=4, **law), _stack(True, granule_gate=4, granule_inner_gate=True, **law)
    fired, loud = _pulses(firing, tokens)
    kept, quiet = _pulses(inner, tokens)
    # The cells fire the same spikes either way.
    assert torch.equal(fired, kept)
    gates = [name for name in loud if name.endswith(".open")]
    bodies = [name for name in loud if name.endswith(".fire") and "granule" in name]
    assert len(gates) == len(bodies) == 2
    assert all(loud[name] > 0 for name in gates) and all(quiet[name] == 0 for name in gates)
    assert all(quiet[name] == loud[name] > 0 for name in bodies)
    # The stack with inner gates sends fewer pulses by those of its gates.
    assert sum(loud.values()) - sum(quiet.values()) == sum(loud[name] for name in gates)


@pytest.mark.parametrize("design", [_RAMP, _INNER])
def test_a_ramp_gated_stack_trains_its_gates_and_its_values(design):
    model = test_lifted_ceiling._stack(**design).train()
    model(_bytes()).float().square().mean().backward()
    for layer in model.granule_layers.values():
        assert layer.gate.grad.abs().sum() > 0 and layer.receive.grad.abs().sum() > 0


@pytest.mark.parametrize("setting", [dict(granule_ramp=8), dict(granule_gate=4, granule_ramp=8, granule_shunt=True),
                                     dict(granule_gate=4, granule_ramp=17), dict(granule_gate=4, granule_ramp=-1),
                                     dict(granule_gate=4, granule_ramp=True), dict(granule_gate=4, granule_ramp=8.0),
                                     dict(granule_ramp=8, granule_after=None), dict(granule_inner_gate=True),
                                     dict(granule_gate=4, granule_inner_gate=1),
                                     dict(granule_gate=4, granule_self_gate=True, granule_inner_gate=True),
                                     dict(granule_inner_gate=True, granule_after=None)])
def test_a_ramp_and_an_inner_gate_belong_to_a_gated_cell(setting):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(test_lifted_ceiling._STACK, **setting)))


def test_the_trainer_trains_ramp_gates_whose_steps_stay_inside(monkeypatch, tmp_path):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / "run", "--granule-gate", "4", "--granule-ramp", "8",
                    "--granule-inner-gate")
    assert record["cfg"]["granule_ramp"] == 8 and record["cfg"]["granule_inner_gate"]
    saved = torch.load(tmp_path / "run" / "model.pt", weights_only=True)
    rebuilt = EventLM(EventLMConfig(**saved["cfg"]))
    rebuilt.load_state_dict(saved["state_dict"])
    assert all(layer.open.dendrite and layer.fire.ramp == 8 for layer in rebuilt.granule_layers.values())
