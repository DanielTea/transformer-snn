"""A silent cell learns from how far its membrane stands below its threshold: the slope far below the threshold."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from snnformer.event_lm import EventLMConfig, _ExcitatoryFire
from test_fine_pool import _FINE, _bytes, _quanta, _stack, _stream


def _slopes(fire, thresholds, **settings):
    """The spikes of ``fire`` at charges given in whole thresholds, and the slope of every count by its charge in
    the cell's own spikes."""
    charge = (thresholds * fire.threshold).requires_grad_()
    spikes = fire(charge, _quanta(4.0), smooth=1, **settings)
    spikes.sum().backward()
    return spikes.detach(), charge.grad * fire.threshold / 16


def test_the_tail_is_off_unless_asked_for_and_adds_no_synapse():
    assert (EventLMConfig.granule_slope_tail, EventLMConfig.granule_slope_tail_local) == (0.0, False)
    plain, tokens = _stack(**_FINE), _bytes()
    for settings in (dict(granule_slope_tail=4.0), dict(granule_slope_tail=4.0, granule_slope_tail_local=True),
                     dict(granule_slope_tail=0.5)):
        model = _stack(**_FINE, **settings)
        # No synapse is added and no spike changes, with a gradient or without one.
        assert set(model.state_dict()) == set(plain.state_dict())
        assert torch.equal(model(tokens), plain(tokens))
        with torch.no_grad():
            assert torch.equal(model(tokens), plain(tokens))
        layer = model.granule_layers["0"]
        # The depth is counted in the cells' own spikes: sixteen fine ones for a whole threshold.
        assert layer.tail == 16 * settings["granule_slope_tail"]
        assert layer.tail_local is settings.get("granule_slope_tail_local", False)
    assert plain.granule_layers["0"].tail == 0


def test_a_silent_cell_learns_by_the_square_of_its_distance_below_zero():
    # Sixty-four quanta to a threshold and four at rest: a cell fires sixteen fine spikes for a threshold of charge.
    fire = _ExcitatoryFire(32, 8, False, quanta_per_threshold=64, capped=True)
    thresholds = torch.tensor([[-24.0, -8.0, -1.0, -0.25, 0.0, 0.25, 2.0]])
    spikes, smooth = _slopes(fire, thresholds)
    for depth in (1.0, 4.0):
        fired, slopes = _slopes(fire, thresholds, tail=16 * depth)
        # The spikes are those of the cell without the tail.
        assert torch.equal(fired, spikes) and spikes.tolist() == [[0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 32.0]]
        # A slope of one from zero charge on, counted once at zero, and one over (1 + distance / depth) squared
        # below it.
        expected = torch.where(thresholds < 0, (1 - thresholds / depth) ** -2, torch.ones(()))
        torch.testing.assert_close(slopes, expected)
    # At the depth a quarter is left, and eight thresholds below zero a cell still learns: a ninth at a depth of
    # four thresholds, where the smooth threshold of one spike has taught nothing since a fifth of a threshold.
    assert slopes[0, 1] == pytest.approx(1 / 9) and smooth[0, :3].abs().max() < 1e-6
    deep = _slopes(fire, torch.tensor([[-4.0]]), tail=64.0)[1]
    assert deep.item() == pytest.approx(0.25)


def test_the_cells_own_charge_takes_the_slope_below_zero_and_the_charge_heard_the_slope_above():
    fire = _ExcitatoryFire(32, 8, False, quanta_per_threshold=64, capped=True)
    thresholds = torch.tensor([[-8.0, -1.0, 0.0, 0.25, 2.0]])
    charge = (thresholds * fire.threshold).requires_grad_()
    own = (thresholds * fire.threshold).requires_grad_()
    quanta = _quanta(4.0).requires_grad_()
    spikes = fire(charge, quanta, smooth=1, tail=64.0, own=own)
    assert spikes.tolist() == [[0.0, 0.0, 0.0, 4.0, 32.0]]
    spikes.sum().backward()
    scale = fire.threshold / 16
    # What the cell hears learns from the charge above zero alone, and the cell's own synapses take the slope from
    # zero down.
    torch.testing.assert_close(charge.grad * scale, torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0]]))
    torch.testing.assert_close(own.grad * scale, torch.tensor([[1 / 9, 0.64, 1.0, 0.0, 0.0]]))
    # The pool's threshold hears only the cells that fire: a count falls as its threshold rises.
    alone = _quanta(4.0).requires_grad_()
    fire((thresholds[:, 3:] * fire.threshold), alone, smooth=1, tail=64.0).sum().backward()
    torch.testing.assert_close(quanta.grad, alone.grad)
    assert quanta.grad.item() < 0


@pytest.mark.parametrize("skips", [None, "level"])
def test_a_local_tail_teaches_the_synapses_of_silent_cells_and_not_the_stream(skips):
    tokens = _bytes()
    found = {}
    for name, settings in (("smooth", dict()), ("all", dict(granule_slope_tail=4.0)),
                           ("local", dict(granule_slope_tail=4.0, granule_slope_tail_local=True))):
        model = _stack(**_FINE, **settings, stack_skips=skips)
        layer = model.granule_layers["0"]
        stream = _stream(model, tokens).detach().requires_grad_()
        # What skip fibres bring, in sixteenths of a spike: here the stream itself at a gain of one.
        heard = stream * 16 if skips else None
        charges, fire = [], layer.fire.forward
        layer.fire.forward = lambda current, *more, **settings: charges.append(current.detach()) or fire(
            current, *more, **settings)
        cells = layer.cells(stream, heard=heard)
        below = charges[0] <= 0
        assert 0.2 < below.float().mean() < 1 and not cells.detach()[below].any()
        # A loss that only the cells at or below zero charge could lower.
        (cells * below).sum().backward()
        found[name] = (cells.detach(), stream.grad, layer.receive.grad, layer.rest.grad)
    spikes = found["smooth"][0]
    assert all(torch.equal(spikes, cells) for cells, *_ in found.values())
    # Under either tail the synapses and rests of those cells learn. With the local one the stream hears nothing
    # of it, and with the other the stream is taught as well.
    for name in ("all", "local"):
        _, stream, receive, rest = found[name]
        assert receive.abs().sum() > 0 and rest.abs().sum() > 0
    assert found["local"][1].abs().sum() == 0 and found["all"][1].abs().sum() > 0
    # The cell's own synapses learn the same from both.
    torch.testing.assert_close(found["local"][2], found["all"][2], rtol=1e-4, atol=1e-7)
    torch.testing.assert_close(found["local"][3], found["all"][3], rtol=1e-4, atol=1e-7)


def test_a_stack_trains_under_a_tail_and_scores_as_before():
    tokens = _bytes()
    plain = _stack(**_FINE)
    for local in (False, True):
        model = _stack(**_FINE, granule_slope_tail=4.0, granule_slope_tail_local=local)
        model.load_state_dict(plain.state_dict())
        output = model(tokens)
        assert torch.equal(output, plain(tokens))
        output.float().square().mean().backward()
        grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert grads and all(torch.isfinite(grad).all() for grad in grads)
        assert model.granule_layers["0"].receive.grad.abs().sum() > 0


@pytest.mark.parametrize("setting", [
    dict(granule_slope_tail=-1.0), dict(granule_slope_tail=65.0), dict(granule_slope_tail=True),
    dict(granule_slope_tail="4"), dict(granule_slope_tail_local=True),
    dict(granule_slope_tail=4.0, granule_slope_tail_local=1),
    dict(granule_slope_tail=4.0, granule_slope=False),
    dict(granule_slope_tail=4.0, granule_slope_width=16),
    dict(granule_slope_tail=4.0, granule_veto=4),
    dict(granule_slope_tail=4.0, granule_slope=False, granule_gate=4, granule_cells=64),
    dict(granule_slope_tail=4.0, granule_branches=4),
])
def test_a_layer_refuses_a_tail_it_cannot_have(setting):
    with pytest.raises(ValueError, match="slope far below the threshold"):
        _stack(**setting)


def test_the_trainer_hands_the_tail_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --granule-slope-tail 4 --granule-slope-tail-local --pool-fine 16 --pool-rest 4 --pool-rate 32
    --output-pool 128 --ctx 16 --bs 2 --steps 2 --warmup 1 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2
    --eval-batches 1 --final-eval-batches 1 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    cfg, layer = model.cfg, model.granule_layers["0"]
    assert (cfg.granule_slope_tail, cfg.granule_slope_tail_local) == (4.0, True)
    assert (layer.tail, layer.tail_local) == (64.0, True)
    # Noisy thresholds teach a cell another slope, so the trainer takes one of the two.
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--cell-noise", "1", "--checkpoint",
                                      str(tmp_path / "noisy.pt"), "--results-dir", str(tmp_path / "results")])
    with pytest.raises(SystemExit):
        train_event_lm.main()
