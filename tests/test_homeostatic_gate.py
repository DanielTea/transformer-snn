"""A ramp gate held in range as cortex holds a neuron: synaptic scaling and intrinsic plasticity together."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import _HOME_RATE, EventLMConfig
from test_teaching_spikes import _bytes, _pieces

# An inner ramp gate that fires four spikes for a threshold of its charge and is wide open at eight thresholds.
_RAMP = dict(granule_gate=4, granule_ramp=8, granule_inner_gate=True)
_HOME = dict(_RAMP, granule_gate_home=0.5)


def _stack(lifted=True, seed=0, **design):
    return test_lifted_ceiling._stack(seed, **dict(design, pool_lifts_ceiling=lifted)).double()


def _layers(model):
    return list(model.granule_layers.values())


def test_the_rule_is_off_unless_asked_for_and_adds_no_synapse():
    assert EventLMConfig.granule_gate_home == 0.0
    plain, home = _stack(**_RAMP), _stack(**_HOME)
    added = set(home.state_dict()) - set(plain.state_dict())
    assert added and all(name.endswith(("gate_total", "gate_shift")) for name in added)
    assert [name for name, _ in home.named_parameters()] == [name for name, _ in plain.named_parameters()]
    assert all(layer.home == 0 for layer in _layers(plain)) and all(layer.home == 0.5 for layer in _layers(home))


@pytest.mark.parametrize("lifted", [False, True])
def test_a_homeostatic_gate_starts_as_the_ramp_gate_it_holds(lifted):
    tokens = _bytes(length=96)
    plain, home = _stack(lifted, **_RAMP).eval(), _stack(lifted, **_HOME).eval()
    home.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.equal(home(tokens), plain(tokens))
    # The first byte took the totals of the gates' synapses as they were.
    for layer in _layers(home):
        torch.testing.assert_close(layer.gate_total, layer.gate.abs().sum(-1), rtol=0, atol=0)


def test_learning_changes_what_a_gate_hears_and_not_how_much():
    tokens = _bytes(length=64)
    model = _stack(**_HOME).eval()
    with torch.no_grad():
        before = model(tokens)
        # Synapses twice as strong are scaled back to the strength the gate started with.
        for layer in _layers(model):
            layer.gate.mul_(2.0)
        assert torch.equal(model(tokens), before)
        # Synapses that point elsewhere change what the gate hears.
        for layer in _layers(model):
            layer.gate.copy_(layer.gate.flip(-1))
        assert not torch.equal(model(tokens), before)


def test_the_local_rule_moves_the_rest_toward_the_set_point():
    layer = _layers(_stack(**_HOME))[0]
    clusters, cells = layer.gate_shift.shape
    quanta = torch.full((2, 5, 1, 1), 4.0, dtype=torch.float64)
    # Under the lifting pool the ramp is wide open at eight thresholds of four gate spikes: 32 spikes.
    wide = torch.full((2, 5, clusters, cells), 32.0, dtype=torch.float64)
    step = _HOME_RATE * 8 * 0.5
    layer.settle(wide, quanta)
    torch.testing.assert_close(layer.gate_shift, torch.full_like(layer.gate_shift, step))
    # A gate at its set point stays where it is, and a shut one is opened by as much.
    layer.settle(wide / 2, quanta)
    torch.testing.assert_close(layer.gate_shift, torch.full_like(layer.gate_shift, step))
    layer.settle(torch.zeros_like(wide), quanta)
    torch.testing.assert_close(layer.gate_shift, torch.zeros_like(layer.gate_shift))
    # The rest moves by the ramp's width at most.
    for _ in range(1000):
        layer.settle(wide, quanta)
    assert torch.equal(layer.gate_shift, torch.full_like(layer.gate_shift, 8.0))


def test_the_rule_moves_nothing_but_while_training():
    tokens = _bytes(length=64)
    model = _stack(**_HOME)
    for training, grad in ((False, False), (False, True), (True, False)):
        model.train(training)
        with torch.set_grad_enabled(grad):
            model(tokens)
        assert all(not layer.gate_shift.any() for layer in _layers(model))
    model.train()
    model(tokens)
    assert all(layer.gate_shift.any() for layer in _layers(model))


@pytest.mark.parametrize("lifted", [False, True])
def test_a_moved_rest_fires_what_its_window_fires_however_it_is_cut(lifted):
    model, tokens = _stack(lifted, **_HOME).eval(), _bytes(length=120)
    with torch.no_grad():
        for layer in _layers(model):
            layer.gate_shift.copy_(torch.linspace(-8, 8, layer.gate_shift.numel()).view_as(layer.gate_shift))
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (120,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 120, (7,) * 17 + (1,), (64, 56)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_a_homeostatic_gate_learns_through_its_scaled_synapses():
    model, tokens = _stack(**_HOME).train(), _bytes(length=64)
    model(tokens).float().square().mean().backward()
    for layer in _layers(model):
        assert layer.gate.grad is not None and torch.isfinite(layer.gate.grad).all()
        assert layer.gate.grad.abs().sum() > 0


@pytest.mark.parametrize("setting", [
    dict(granule_gate_home=0.5), dict(granule_gate=4, granule_gate_home=0.5),
    dict(_RAMP, granule_gate_home=1.0), dict(_RAMP, granule_gate_home=-0.1), dict(_RAMP, granule_gate_home=True),
    dict(granule_gate=4, granule_ramp=8, granule_self_gate=True, granule_gate_home=0.5),
])
def test_a_layer_refuses_a_homeostatic_gate_it_cannot_have(setting):
    with pytest.raises(ValueError, match="homeostatic gate"):
        _stack(**setting)


def test_the_trainer_hands_the_set_point_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --granule-gate 4 --granule-ramp 8 --granule-inner-gate --granule-gate-home 0.5 --pool-fine 16
    --pool-rest 4 --pool-rate 32 --output-pool 128 --pool-lifts-ceiling --ctx 16 --bs 2 --steps 2 --warmup 1
    --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1 --final-eval-batches 1
    --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    layer = model.granule_layers["0"]
    assert model.cfg.granule_gate_home == 0.5 and layer.home == 0.5
    # Two updates of training moved the rests, and the checkpoint keeps where they stand.
    assert layer.gate_shift.any() and layer.gate_total.any()
