"""Cubic cells: power-law cells whose count grows with the cube, round(k^3 / power^2) spikes for k thresholds."""

import math

import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import _CUBE_STEP_BITS, EventLM, EventLMConfig, _ExcitatoryFire
from test_teaching_spikes import _bytes, _pieces

# A whole threshold of 256 ticks counts 64 quanta (a quantum of 4 ticks), and the membrane holds 128 whole thresholds.
_THRESHOLD = 256


def _fire(power, lifted=False, exponent=3):
    fire = _ExcitatoryFire(128, 8, False, quanta_per_threshold=64, capped=True)
    fire.lifted, fire.power, fire.exponent = lifted, power, exponent
    return fire


def _stack(seed=0, **design):
    return test_lifted_ceiling._stack(seed, **design).double()


def _rule(charge, threshold, power, limit, bound=None):
    """The count by the record's rule, in Python's whole numbers: k in 4,096ths of a threshold, cubed, over power^2."""
    charge = max(0, charge if bound is None else min(charge, bound))
    steps = (charge << _CUBE_STEP_BITS) // threshold
    unit = 1 << (3 * _CUBE_STEP_BITS)
    return min((2 * steps ** 3 + power * power * unit) // (2 * power * power * unit), limit)


def test_cubic_cells_are_off_unless_asked_for():
    assert EventLMConfig.granule_power_exponent == 2 and _ExcitatoryFire.exponent == 2
    assert all(layer.fire.exponent == 2 for layer in _stack(granule_power=8).granule_layers.values())
    cubic = _stack(granule_power=8, granule_power_exponent=3)
    assert all(layer.fire.exponent == 3 and layer.fire.power == 8 for layer in cubic.granule_layers.values())


@pytest.mark.parametrize("lifted", [False, True])
def test_a_cell_fires_the_rounded_cube_of_its_thresholds_over_the_power_squared(lifted):
    fire = _fire(8, lifted)
    # k thresholds of charge: k^3 / 64, rounded half up, nothing below the rest, at most the cap of 128.
    thresholds = torch.tensor([-3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 16.0, 20.0, 40.0], dtype=torch.float64)
    counts = fire(thresholds * _THRESHOLD, torch.tensor(64.0), smooth=1)
    assert counts.tolist() == [0, 0, 0, 0, 0, 1, 2, 8, 16, 64, 125, 128]
    # At eight thresholds it fires as a plain cell and a square cell do.
    plain = _fire(0, lifted)(torch.tensor([8.0 * _THRESHOLD], dtype=torch.float64), torch.tensor(64.0))
    square = _fire(8, lifted, exponent=2)(torch.tensor([8.0 * _THRESHOLD], dtype=torch.float64), torch.tensor(64.0))
    assert counts[7] == plain == square


@pytest.mark.parametrize("units", [64, 37, 100, 160])
@pytest.mark.parametrize("power", [8, 3, 64])
def test_the_counts_follow_the_whole_number_rule_at_any_pool_threshold(units, power):
    fire = _fire(power)
    threshold = units * (_THRESHOLD // 64)
    limit = min(128 * 64 // units, 128)
    charge = torch.arange(-500, 90 * threshold, 13, dtype=torch.int64)
    counts = fire(charge.double(), torch.tensor(float(units)), smooth=1).long()
    expected = [_rule(c, threshold, power, limit, bound=_THRESHOLD * 128) for c in charge.tolist()]
    assert counts.tolist() == expected


def test_whole_numbers_in_give_the_same_counts_as_floats():
    for power, lifted in ((8, False), (3, True), (64, True)):
        fire = _fire(power, lifted)
        charge = torch.arange(-1000, 60 * _THRESHOLD, 61)
        floats = fire(charge.double(), torch.tensor(64.0), smooth=1)
        assert torch.equal(fire(charge.int(), torch.tensor(64)).long(), floats.long())


def test_a_huge_charge_under_a_lifted_ceiling_fires_the_cap_and_stays_within_64_bits():
    fire = _fire(64, lifted=True)
    floats = fire(torch.tensor([1e15, 3.0e9, 1e30], dtype=torch.float64), torch.tensor(64.0), smooth=1)
    assert floats.tolist() == [128, 128, 128]
    whole = fire(torch.tensor([1 << 50, (1 << 40) + 7], dtype=torch.int64), torch.tensor(64))
    assert whole.tolist() == [128, 128]


def test_the_slope_is_that_of_the_cube_of_a_smooth_threshold_and_flat_past_the_cap():
    fire = _fire(8)
    thresholds = torch.tensor([-2.0, 0.0, 1.5, 6.0, 15.0, 21.0, 50.0], dtype=torch.float64, requires_grad=True)
    fire(thresholds * _THRESHOLD, torch.tensor(64.0), smooth=1).sum().backward()
    flat = (8 * 8 * 128) ** (1 / 3)
    for value, slope in zip(thresholds.detach().tolist(), thresholds.grad.tolist()):
        rising = math.log1p(math.exp(value))
        expected = 3 * rising * rising / (1 + math.exp(-value)) / 64 if rising < flat else 0.0
        assert slope == pytest.approx(expected, rel=1e-9, abs=1e-12)
    # A silent cell still learns, and the slope rises with the charge up to the cap.
    assert 0 < thresholds.grad[0] < thresholds.grad[1] < thresholds.grad[2] < thresholds.grad[3] < thresholds.grad[4]


def test_a_cubic_stack_has_the_plain_stacks_synapses_and_fires_otherwise_than_the_square():
    square, cubic = _stack(granule_power=8), _stack(granule_power=8, granule_power_exponent=3)
    assert {name: tuple(value.shape) for name, value in square.state_dict().items()} == {
        name: tuple(value.shape) for name, value in cubic.state_dict().items()}
    cubic.load_state_dict(square.state_dict())
    tokens = _bytes(length=60)
    with torch.no_grad():
        assert not torch.equal(cubic.eval()(tokens), square.eval()(tokens))


@pytest.mark.parametrize("design", [dict(), dict(pool_lifts_ceiling=False), dict(stack_loop=(0, 1, 2))])
def test_a_cubic_stack_fires_what_its_window_fires_however_it_is_cut(design):
    model, tokens = _stack(**design, granule_power=4, granule_power_exponent=3).eval(), _bytes(length=90)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (90,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 90, (7,) * 12 + (6,), (45, 45), (13, 77)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_cubic_cells_learn_through_their_slope():
    model, tokens = _stack(granule_power=8, granule_power_exponent=3).train(), _bytes(length=64)
    model(tokens).float().square().mean().backward()
    for layer in model.granule_layers.values():
        grads = [p.grad for p in layer.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("design", [dict(granule_power=8, granule_power_exponent=1),
                                    dict(granule_power=8, granule_power_exponent=4),
                                    dict(granule_power=8, granule_power_exponent=True),
                                    dict(granule_power_exponent=3)])
def test_the_power_is_two_or_three_and_needs_power_law_cells(design):
    with pytest.raises(ValueError, match="power-law cells"):
        EventLM(EventLMConfig(**dict(test_lifted_ceiling._STACK, **design)))
