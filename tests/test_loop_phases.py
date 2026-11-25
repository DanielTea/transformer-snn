"""Phases of a loop: the later passes of a looped level have their own rests and read gains beside the shared synapses."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import EventLMConfig
from test_teaching_spikes import _bytes, _pieces

# Two levels, both run three times: the passes visit levels 0, 1, 0, 1, 0, 1.
_LOOP = dict(stack_loop=(0, 1, 3))


def _stack(seed=0, **design):
    return test_lifted_ceiling._stack(seed, **design).double()


def _offsets(model):
    return list(model.phase_rests.values()) + list(model.phase_gains.values())


def _move(model):
    """Give every pass's offsets values that change what it fires."""
    with torch.no_grad():
        for rest in model.phase_rests.values():
            rest.copy_(torch.linspace(0, 3, rest.numel()).view_as(rest))
        for gain in model.phase_gains.values():
            gain.fill_(0.05)


def test_phases_are_off_unless_asked_for_and_belong_to_the_later_passes():
    assert EventLMConfig.stack_loop_phases is False
    plain, phased = _stack(**_LOOP), _stack(**_LOOP, stack_loop_phases=True)
    assert plain.phase_rests is None and plain.phase_gains is None
    # The first pass of each level has none; passes 2 to 5 do.
    assert sorted(phased.phase_rests) == sorted(phased.phase_gains) == ["2", "3", "4", "5"]
    added = set(phased.state_dict()) - set(plain.state_dict())
    assert added and all(name.startswith(("phase_rests.", "phase_gains.")) for name in added)
    assert all(not offset.any() for offset in _offsets(phased))


def test_phases_at_zero_are_the_loop_without_them():
    tokens = _bytes(length=60)
    plain, phased = _stack(**_LOOP).eval(), _stack(**_LOOP, stack_loop_phases=True).eval()
    phased.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        assert torch.equal(phased(tokens), plain(tokens))
        _move(phased)
        assert not torch.equal(phased(tokens), plain(tokens))


@pytest.mark.parametrize("design", [dict(), dict(pool_lifts_ceiling=False), dict(stack_sleep=16)])
def test_a_phased_loop_fires_what_its_window_fires_however_it_is_cut(design):
    model, tokens = _stack(**_LOOP, **design, stack_loop_phases=True).eval(), _bytes(length=90)
    _move(model)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (90,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 90, (7,) * 12 + (6,), (45, 45), (13, 77)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


def test_every_pass_learns_its_own_offsets():
    model, tokens = _stack(**_LOOP, stack_loop_phases=True).train(), _bytes(length=64)
    model(tokens).float().square().mean().backward()
    for offset in _offsets(model):
        assert offset.grad is not None and torch.isfinite(offset.grad).all() and offset.grad.abs().sum() > 0


def test_a_rest_offset_keeps_the_cells_rest_in_range():
    model = _stack(**_LOOP, stack_loop_phases=True)
    layer = model.granule_layers["2"]
    with torch.no_grad():
        model.phase_rests["2"].fill_(100.0)
    assert layer.rest_quarters(model.phase_rests["2"]).max() <= 16 * 4


@pytest.mark.parametrize("setting", [dict(stack_loop_phases=True), dict(stack_loop=(0, 1, 1), stack_loop_phases=True),
                                     dict(_LOOP, stack_loop_phases=1)])
def test_a_stack_refuses_phases_it_cannot_have(setting):
    with pytest.raises(ValueError, match="phases"):
        _stack(**setting)


def test_the_trainer_hands_the_phases_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 2 --spike-resolution 16
    --router-charge-bits 8 --read-after 0,1 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0,1 --granule-clusters 1 --granule-fan 32
    --granule-cells 16 --granule-slope --pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128
    --pool-lifts-ceiling --stack-loop 0,1,2 --stack-loop-phases --ctx 16 --bs 2 --steps 2 --warmup 1 --lr 0.003
    --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1 --final-eval-batches 1 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    assert model.cfg.stack_loop_phases is True and sorted(model.phase_rests) == ["2", "3"]
    # Two updates moved the offsets.
    assert any(offset.any() for offset in _offsets(model))
