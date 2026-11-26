"""Power-law cells: a plain threshold cell fires round(k * k / power) spikes for k thresholds of charge above its rest."""

import math

import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import EventLM, EventLMConfig, _ExcitatoryFire
from test_teaching_spikes import _bytes, _pieces

# A whole threshold of 256 ticks counts 64 quanta (a quantum of 4 ticks), and the membrane holds 128 whole thresholds.
_THRESHOLD = 256


def _fire(power, lifted=False):
    fire = _ExcitatoryFire(128, 8, False, quanta_per_threshold=64, capped=True)
    fire.lifted, fire.power = lifted, power
    return fire


def _stack(seed=0, **design):
    return test_lifted_ceiling._stack(seed, **design).double()


def test_power_law_cells_are_off_unless_asked_for():
    assert EventLMConfig.granule_power == 0 and _ExcitatoryFire.power == 0
    assert all(layer.fire.power == 0 for layer in _stack().granule_layers.values())
    assert all(layer.fire.power == 8 for layer in _stack(granule_power=8).granule_layers.values())


@pytest.mark.parametrize("lifted", [False, True])
def test_a_cell_fires_the_rounded_square_of_its_thresholds_over_the_power(lifted):
    fire = _fire(8, lifted)
    # k thresholds of charge: k * k / 8, rounded half up, and nothing below the rest.
    thresholds = torch.tensor([-3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 8.0, 12.0, 31.0, 32.0, 40.0], dtype=torch.float64)
    counts = fire(thresholds * _THRESHOLD, torch.tensor(64.0), smooth=1)
    assert counts.tolist() == [0, 0, 0, 1, 1, 2, 8, 18, 120, 128, 128]
    # At eight thresholds it fires as a plain cell does.
    assert counts[6] == _fire(0, lifted)(torch.tensor([8.0 * _THRESHOLD], dtype=torch.float64), torch.tensor(64.0))


def test_a_raised_threshold_lowers_the_count_and_the_cap_as_for_a_plain_cell():
    fire, plain = _fire(4), _fire(0)
    # A pool that has raised the threshold to 160 quanta: 640 ticks, and at most 128 * 64 // 160 = 51 spikes.
    charge = torch.arange(0, 60 * 640, 97, dtype=torch.float64)
    counts = fire(charge, torch.tensor(160.0), smooth=1)
    whole = charge.long()
    expected = ((2 * whole * whole + 4 * 640 * 640) // (2 * 4 * 640 * 640)).clamp(max=51)
    assert torch.equal(counts.long(), expected)
    assert counts.max() == plain(charge, torch.tensor(160.0)).max() == 51


def test_whole_numbers_in_give_the_same_counts_as_floats():
    fire = _fire(8)
    charge = torch.arange(-1000, 40 * _THRESHOLD, 61)
    floats = fire(charge.double(), torch.tensor(64.0), smooth=1)
    assert torch.equal(fire(charge.int(), torch.tensor(64)).long(), floats.long())


def test_the_slope_is_that_of_the_square_of_a_smooth_threshold_and_flat_past_the_cap():
    fire = _fire(8)
    thresholds = torch.tensor([-2.0, 0.0, 1.5, 6.0, 20.0, 33.0, 50.0], dtype=torch.float64, requires_grad=True)
    fire(thresholds * _THRESHOLD, torch.tensor(64.0), smooth=1).sum().backward()
    for value, slope in zip(thresholds.detach().tolist(), thresholds.grad.tolist()):
        rising = math.log1p(math.exp(value))
        expected = 2 * rising / (1 + math.exp(-value)) / 8 if rising < math.sqrt(8 * 128) else 0.0
        assert slope == pytest.approx(expected, rel=1e-9, abs=1e-12)
    # A silent cell still learns, and the slope rises with the charge up to the cap.
    assert 0 < thresholds.grad[0] < thresholds.grad[1] < thresholds.grad[2] < thresholds.grad[3] < thresholds.grad[4]


def test_a_power_law_stack_has_the_plain_stacks_synapses_and_fires_otherwise():
    plain, power = _stack(), _stack(granule_power=8)
    assert {name: tuple(value.shape) for name, value in plain.state_dict().items()} == {
        name: tuple(value.shape) for name, value in power.state_dict().items()}
    power.load_state_dict(plain.state_dict())
    tokens = _bytes(length=60)
    with torch.no_grad():
        assert not torch.equal(power.eval()(tokens), plain.eval()(tokens))


@pytest.mark.parametrize("design", [dict(), dict(pool_lifts_ceiling=False), dict(stack_loop=(0, 1, 2))])
def test_a_power_law_stack_fires_what_its_window_fires_however_it_is_cut(design):
    model, tokens = _stack(**design, granule_power=4).eval(), _bytes(length=90)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (90,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 90, (7,) * 12 + (6,), (45, 45), (13, 77)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_power_law_cells_learn_through_their_slope():
    model, tokens = _stack(granule_power=8).train(), _bytes(length=64)
    model(tokens).float().square().mean().backward()
    for layer in model.granule_layers.values():
        grads = [p.grad for p in layer.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("design", [dict(granule_power=-1), dict(granule_power=65), dict(granule_power=True),
                                    dict(granule_power=8, granule_gate=4),
                                    dict(granule_power=8, granule_slope_tail=4.0)])
def test_power_law_cells_are_plain_threshold_cells_with_a_power_in_range(design):
    with pytest.raises(ValueError, match="power-law cells"):
        EventLM(EventLMConfig(**dict(test_lifted_ceiling._STACK, **design)))


def test_power_law_cells_need_cell_layers():
    settings = {k: v for k, v in test_lifted_ceiling._STACK.items() if k != "granule_after"}
    with pytest.raises(ValueError, match="power-law cells"):
        EventLM(EventLMConfig(**settings, granule_power=8))
