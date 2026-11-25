"""A gated cell that multiplies two drives of either sign: the signed gate."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import EventLMConfig
from test_teaching_spikes import _bytes, _pieces

# An inner multiplying gate of four steps to a threshold, signed.
_PLAIN = dict(granule_gate=4, granule_inner_gate=True)
_SIGNED = dict(_PLAIN, granule_gate_signed=True)


def _stack(lifted=True, seed=0, **design):
    return test_lifted_ceiling._stack(seed, **dict(design, pool_lifts_ceiling=lifted)).double()


def _layers(model):
    return list(model.granule_layers.values())


def test_the_signed_gate_is_off_unless_asked_for_and_adds_no_synapse():
    assert EventLMConfig.granule_gate_signed is False
    plain, signed = _stack(**_PLAIN), _stack(**_SIGNED)
    added = set(signed.state_dict()) - set(plain.state_dict())
    assert added and all(name.endswith("gate_total") for name in added)
    assert [name for name, _ in signed.named_parameters()] == [name for name, _ in plain.named_parameters()]
    assert all(layer.signed and layer.scaled and not layer.home for layer in _layers(signed))


@pytest.mark.parametrize("lifted", [False, True])
def test_the_body_fires_the_product_of_two_signed_numbers(lifted):
    model, tokens = _stack(lifted, **_SIGNED).eval(), _bytes(length=96)
    seen = []

    def record(_module, args, kwargs, counts):
        seen.append((args[0].detach(), kwargs["opened"].detach(), counts.detach()))

    handles = [layer.fire.register_forward_hook(record, with_kwargs=True) for layer in _layers(model)]
    with torch.no_grad():
        model(tokens)
    for handle in handles:
        handle.remove()
    for charge, opened, counts in seen:
        # The gate fires on both rails, and the body on the rail of the product's sign.
        assert (opened > 0).any() and (opened < 0).any()
        speaking = counts != 0
        assert speaking.any() and (counts < 0).any() and (counts > 0).any()
        assert torch.equal(counts[speaking].sign(), (charge * opened)[speaking].sign())
        assert not counts[opened == 0].any()


@pytest.mark.parametrize("lifted", [False, True])
def test_a_signed_gate_fires_what_its_window_fires_however_it_is_cut(lifted):
    model, tokens = _stack(lifted, **_SIGNED).eval(), _bytes(length=120)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (120,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 120, (7,) * 17 + (1,), (64, 56)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_a_signed_gate_hears_with_the_strength_it_started_with():
    tokens = _bytes(length=64)
    model = _stack(**_SIGNED).eval()
    with torch.no_grad():
        before = model(tokens)
        for layer in _layers(model):
            layer.gate.mul_(2.0)
        assert torch.equal(model(tokens), before)


def test_a_signed_gate_learns_through_both_of_its_drives():
    model, tokens = _stack(**_SIGNED).train(), _bytes(length=64)
    model(tokens).float().square().mean().backward()
    for layer in _layers(model):
        for weights in (layer.gate, layer.receive):
            assert weights.grad is not None and torch.isfinite(weights.grad).all() and weights.grad.abs().sum() > 0


@pytest.mark.parametrize("setting", [
    dict(granule_gate_signed=True), dict(granule_gate=4, granule_gate_signed=True),
    dict(_SIGNED, granule_ramp=8), dict(_SIGNED, granule_shunt=True),
    dict(granule_gate=4, granule_self_gate=True, granule_gate_signed=True),
    dict(_PLAIN, granule_gate_signed=1),
])
def test_a_layer_refuses_a_signed_gate_it_cannot_have(setting):
    with pytest.raises(ValueError, match="signed gate|inner gate"):
        _stack(**setting)


def test_the_trainer_hands_the_signed_gate_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --granule-gate 4 --granule-inner-gate --granule-gate-signed --pool-fine 16 --pool-rest 4
    --pool-rate 32 --output-pool 128 --pool-lifts-ceiling --ctx 16 --bs 2 --steps 2 --warmup 1 --lr 0.003 --seed 3
    --eval-every 2 --eval-bs 2 --eval-batches 1 --final-eval-batches 1 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    layer = model.granule_layers["0"]
    assert model.cfg.granule_gate_signed is True and layer.signed and layer.gate_total.any()
