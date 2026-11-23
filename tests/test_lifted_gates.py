"""Gated cells under a pool that lifts the ceiling: neither the gate nor the body is bounded by its membrane."""

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _GatedFire
from test_lifted_ceiling import _STACK, _bytes, _stack
from test_teaching_spikes import _pieces

_SHUNT = dict(granule_gate=4, granule_shunt=True)
_SELF = dict(granule_gate=4, granule_self_gate=True, granule_shunt=True)


def _body(lifted, shunt=True):
    # A whole threshold of 256 ticks counts 64 quanta, the membrane holds 128 whole thresholds, 128 spikes at most.
    body = _GatedFire(128, 8, 4, 64, shunt=shunt)
    body.lifted = lifted
    return body


@pytest.mark.parametrize("shunt", [True, False])
@pytest.mark.parametrize("quanta", [4, 16, 64, 100])
def test_a_lifted_body_fires_as_before_below_its_old_ceiling(shunt, quanta):
    generator = torch.Generator().manual_seed(quanta)
    bound = 256 * 128
    charge = torch.randint(-bound, bound + 1, (4096,), generator=generator).float()
    gate = torch.randint(0, 65, (4096,), generator=generator).float()
    pool = torch.tensor([float(quanta)])
    old, new = _body(False, shunt)(charge, pool, opened=gate), _body(True, shunt)(charge, pool, opened=gate)
    # Where the old body stayed below its largest count, the lifted one fires the same.
    free = old.abs() < old.abs().max()
    assert torch.equal(new[free], old[free])
    assert torch.equal(new, new.round()) and new.abs().max() <= 128


def test_a_lifted_body_counts_past_its_membrane_under_a_loud_pool():
    quanta = torch.tensor([512.0])
    # Under 512 quanta a fine spike at a shunt of k = 60 takes 2048 * 64 / 60 ticks of collected charge.
    gate = torch.full((6,), 60.0)
    charge = torch.tensor([0.0, 10.0, 40.0, 100.0, 1000.0, 5000.0]) * 2048
    old, new = _body(False)(charge, quanta, opened=gate), _body(True)(charge, quanta, opened=gate)
    assert old.max() <= 16 and (new > old).any() and new.max() == 128
    expected = torch.floor(charge * 60 / (512 * 4 * 64) + 0.5).clamp(max=128)
    assert torch.equal(new, expected)
    # Whole numbers of charge fire the same counts.
    assert torch.equal(_body(True)(charge.to(torch.int32), quanta.to(torch.int32), opened=gate.to(torch.int32)),
                       new.to(torch.int32))


def test_a_lifted_body_learns_through_its_value_and_its_gate_up_to_its_largest_count():
    quanta = torch.tensor([512.0])
    charge = torch.tensor([10.0 * 2048, 6000.0 * 2048], requires_grad=True)
    gate = torch.tensor([8.0, 8.0], requires_grad=True)
    _body(True)(charge, quanta, opened=gate).sum().backward()
    assert charge.grad[0] > 0 and gate.grad[0] > 0
    # Past its largest count a body learns nothing.
    assert charge.grad[1] == 0 and gate.grad[1] == 0


@pytest.mark.parametrize("design", [_SHUNT, _SELF])
def test_a_lifted_gated_stack_lifts_its_gates_and_bodies_and_streams_exactly(design):
    model = _stack(**design).double().eval()
    lifted = {name for name, module in model.named_modules() if getattr(module, "lifted", False)}
    assert {f"granule_layers.{level}.{part}" for level in (0, 1) for part in ("fire", "open")} <= lifted
    tokens = _bytes(length=120)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (120,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 120, (7,) * 17 + (1,), (64, 56)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_a_lifted_gated_stack_trains_its_gates_and_values():
    model = _stack(**_SHUNT).train()
    model(_bytes()).float().square().mean().backward()
    for layer in model.granule_layers.values():
        assert layer.gate.grad.abs().sum() > 0 and layer.receive.grad.abs().sum() > 0


def test_a_veto_still_refuses_a_lifting_pool():
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, granule_veto=4)))
