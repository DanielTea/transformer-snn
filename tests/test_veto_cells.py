"""A cell with a veto dendrite fires for what one view of the stream collects unless another view forbids it."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _ExcitatoryFire, _IntegrateFire, _VetoedFire, emitted_pulses

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=64, granule_slope=True, granule_veto=4)
# Sixty-four quanta to a threshold with four at rest, a pool spike for every eight pulses of 64 channels, and a pool
# at the output.
_FINE = dict(pool_fine=16, pool_rest=4, pool_rate=32, output_pool=128)


def _stack(seed=0, grown=True, veto=0.0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    if grown:
        # Reads and cell layers start silent. Give them output synapses, and the veto dendrites theirs.
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for read in model.state_reads.values():
                read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
            for layer in model.granule_layers.values():
                layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
                if veto:
                    layer.veto.copy_(torch.randn(layer.veto.shape, generator=generator) * veto)
    return model


def _bytes(seed=2, length=80):
    return torch.randint(0, 256, (2, length), generator=torch.Generator().manual_seed(seed))


def _stream(model, tokens):
    return model.input_emit(model.input(tokens) * model.cfg.resolution)


def _quanta(count):
    return torch.full((1,), float(count))


def test_a_cell_has_a_veto_dendrite_beside_its_own():
    plain, vetoed = _stack(granule_veto=0, granule_cells=96), _stack()
    extra = set(vetoed.state_dict()) - set(plain.state_dict())
    assert extra == {f"granule_layers.{level}.veto" for level in _LEVELS}
    layer = vetoed.granule_layers["0"]
    assert layer.veto.shape == layer.receive.shape == (1, 64, 64) and layer.rest.shape == (1, 64, 1)
    # The veto's synapses start silent.
    assert not layer.veto.any()
    # Three rows of synapses a cell, where a plain cell has two: 64 cells with a veto cost what 96 plain cells cost.
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    assert count(vetoed) - count(plain) == len(_LEVELS) * (64 - 96)


def test_the_body_fires_for_the_share_that_the_veto_leaves():
    fire = _VetoedFire(128, 8, 4)
    # At rest the pool gives four quanta, and a threshold is 256 ticks.
    charge = torch.tensor([512.0, -512.0, 300.0, 127.0, 1000.0])
    steps = lambda count: torch.full((5,), float(count))  # noqa: E731
    # With no veto the body fires as a plain cell: a spike for every threshold of charge, rounded, and none below.
    plain = _ExcitatoryFire(128, 8, False, capped=True)
    assert fire(charge, _quanta(4), vetoed=steps(0)).tolist() == plain(charge, _quanta(4)).tolist() == [2, 0, 1, 0, 4]
    # Every step keeps a quarter of the charge from the body, and the fourth shuts the cell.
    assert fire(charge, _quanta(4), vetoed=steps(1)).tolist() == [2, 0, 1, 0, 3]
    assert fire(charge, _quanta(4), vetoed=steps(2)).tolist() == [1, 0, 1, 0, 2]
    assert fire(charge, _quanta(4), vetoed=steps(3)).tolist() == [1, 0, 0, 0, 1]
    assert fire(charge, _quanta(4), vetoed=steps(4)).tolist() == [0, 0, 0, 0, 0]
    # The pool's inhibition raises the threshold: with eight quanta the same charge counts half.
    assert fire(charge, _quanta(8), vetoed=steps(0)).tolist() == [1, 0, 1, 0, 2]
    # The collected charge is bounded like a plain cell's, and the count ends at the cell's largest.
    assert fire(torch.tensor([10.0 ** 7]), _quanta(4), vetoed=torch.zeros(1)).tolist() == [128]
    assert fire(torch.tensor([10.0 ** 7]), _quanta(4), vetoed=torch.full((1,), 2.0)).tolist() == [64]
    # Whole numbers in, whole numbers out.
    whole = fire(charge.to(torch.int32), _quanta(4).to(torch.int32), vetoed=torch.full((5,), 1, dtype=torch.int32))
    assert whole.dtype == torch.int32 and whole.tolist() == [2, 0, 1, 0, 3]
    # The share that passes is a whole number of ticks that FP32 holds exactly.
    assert (fire.threshold, fire.steps) == (1024, 4) and fire.threshold * fire.cap < 1 << 24


def test_a_fine_body_counts_the_share_in_fine_spikes():
    fire = _VetoedFire(128, 8, 4, 64)
    plain = _ExcitatoryFire(128, 8, False, quanta_per_threshold=64, capped=True)
    charge = torch.tensor([512.0, 300.0, 127.0, 9.0, -40.0])
    # Sixty-four quanta to a threshold and four at rest: sixteen fine spikes for a threshold of charge.
    assert fire(charge, _quanta(4), vetoed=torch.zeros(5)).tolist() == plain(charge, _quanta(4)).tolist()
    assert fire(charge, _quanta(4), vetoed=torch.zeros(5)).tolist() == [32, 19, 8, 1, 0]
    assert fire(charge, _quanta(4), vetoed=torch.full((5,), 2.0)).tolist() == [16, 9, 4, 0, 0]
    for quanta in (4, 7, 23, 64, 300):
        for vetoed in range(5):
            counts = fire(charge, _quanta(quanta), vetoed=torch.full((5,), float(vetoed)))
            # The count is the share of the charge over the present threshold, rounded half up.
            expected = ((charge.clamp_min(0) * (4 - vetoed) / (16 * quanta)) + 0.5).floor()
            assert torch.equal(counts, expected.clamp(max=128 * 64 // quanta))


def test_the_body_learns_through_both_dendrites():
    fire = _VetoedFire(128, 8, 4)
    charge = torch.tensor([512.0, 300.0, -300.0, 600.0], requires_grad=True)
    vetoed = torch.tensor([0.0, 2.0, 1.0, 4.0], requires_grad=True)
    fire(charge, _quanta(4), vetoed=vetoed, smooth=1).sum().backward()
    slope = charge.grad * 256
    # The cell's own dendrite learns as a plain cell's times the share that passes: all of it, half, and nothing
    # where the cell is shut.
    assert abs(slope[0] - 1) < 0.1 and 0.4 < slope[1] < 0.65 and slope[3] == 0
    # A step of veto takes a quarter of what the cell would have fired, and nothing from a cell that is silent
    # without it.
    assert abs(vetoed.grad[0] + 2 / 4) < 0.02 and abs(vetoed.grad[1] + 300 / 256 / 4) < 0.05
    assert vetoed.grad[2] == 0 and abs(vetoed.grad[3] + 600 / 256 / 4) < 0.02
    # With no veto the slope is the plain cell's own.
    plain = _ExcitatoryFire(128, 8, False, capped=True)
    for smooth in (1, 4):
        first = torch.tensor([512.0, 300.0, -40.0, -300.0, 100.0], requires_grad=True)
        second = first.detach().clone().requires_grad_()
        fire(first, _quanta(4), vetoed=torch.zeros(5), smooth=smooth).sum().backward()
        plain(second, _quanta(4), smooth=smooth).sum().backward()
        torch.testing.assert_close(first.grad, second.grad)


def test_the_veto_dendrite_counts_its_steps_over_one_whole_spike():
    for settings in (dict(), _FINE):
        model = _stack(**settings)
        layer = model.granule_layers["0"]
        shut = layer.shut
        assert shut.dendrite and not layer.fire.dendrite and shut.capped == 4
        # At rest a whole spike is 256 ticks of charge, in one spike or in sixteen fine ones. A step of the veto is
        # a quarter of it, counted to the nearest step.
        rest = _quanta(model.cfg.pool_rest_quanta)
        charge = torch.tensor([-256.0, 0.0, 31.0, 32.0, 96.0, 160.0, 223.0, 224.0, 9 * 256.0])
        assert shut(charge, rest).tolist() == [0, 0, 0, 1, 2, 3, 3, 4, 4]
        # With twice the quanta the veto needs twice the charge.
        assert shut(2 * charge, 2 * rest).tolist() == [0, 0, 0, 1, 2, 3, 3, 4, 4]
        # It reaches its last step under the strongest inhibition of the pool.
        most = layer.rest_quanta + layer.pool.cap
        assert shut(torch.tensor([64.0 * most]), _quanta(most)).tolist() == [4]
        assert shut.threshold * shut.cap < 1 << 24
    # The veto learns below its first step, as a silent cell does.
    charge = torch.tensor([-40.0, 0.0, 30.0], requires_grad=True)
    _stack().granule_layers["0"].shut(charge, _quanta(4), smooth=4).sum().backward()
    assert (charge.grad > 0).all()


@pytest.mark.parametrize("settings", [dict(), _FINE, dict(_FINE, granule_slope_width=16),
                                      dict(granule_fan=128, granule_delays=4)])
def test_cells_whose_veto_is_silent_are_plain_cells(settings):
    plain, vetoed, tokens = _stack(granule_veto=0, **settings), _stack(**settings), _bytes()
    vetoed.load_state_dict(plain.state_dict(), strict=False)
    first, second = plain(tokens), vetoed(tokens)
    assert torch.equal(first, second)
    for output in (first, second):
        output.float().square().mean().backward()
    for name, parameter in plain.named_parameters():
        torch.testing.assert_close(parameter.grad, vetoed.get_parameter(name).grad)
    # The silent veto learns from the cells that fire.
    for layer in vetoed.granule_layers.values():
        assert layer.veto.grad.abs().sum() > 0


def test_a_veto_closes_cells_that_would_have_fired():
    model, tokens = _stack(**_FINE), _bytes()
    layer = model.granule_layers["0"]
    stream = _stream(model, tokens)
    open_cells = layer.cells(stream)
    assert (open_cells > 0).any()
    with torch.no_grad():
        # A veto dendrite that hears the fibres as the cell's own dendrite does closes the cell where it is driven.
        layer.veto.copy_(layer.receive * 8)
    closed = layer.cells(stream)
    assert (closed <= open_cells).all() and closed.sum() < open_cells.sum() / 2
    with torch.no_grad():
        layer.veto.copy_(-layer.receive * 8)
    # A veto dendrite that is charged only where its cell is not driven changes no spike.
    assert torch.equal(layer.cells(stream), open_cells)


def test_a_stack_of_cells_with_vetoes_fires_whole_spikes_and_sees_no_later_byte():
    for settings in (dict(), _FINE):
        model, tokens = _stack(veto=0.3, **settings), _bytes()
        changed = tokens.clone()
        changed[:, 50:] = _bytes(seed=7)[:, 50:]
        first, second = model(tokens), model(changed)
        assert torch.equal(first, first.round())
        assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
        layer = model.granule_layers["0"]
        stream = _stream(model, tokens)
        cells = layer.cells(stream)
        assert torch.equal(cells, cells.round()) and (cells >= 0).all()
        # The random vetoes are at work: the cells differ from those of a silent veto.
        with torch.no_grad():
            kept = layer.veto.clone()
            layer.veto.zero_()
            plain = layer.cells(stream)
            layer.veto.copy_(kept)
        assert not torch.equal(cells, plain)


def test_every_synapse_of_a_stack_with_vetoes_learns():
    for settings in (dict(), _FINE):
        model = _stack(veto=0.3, **settings)
        model(_bytes()).float().square().mean().backward()
        assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_cells_with_vetoes_stay_within_the_range_of_exact_sums():
    model = _stack(veto=0.3, **_FINE)
    model._check_accumulator()
    layer = model.granule_layers["1"]
    own, _ = layer.accumulator_bounds()
    with torch.no_grad():
        # A cell's own dendrite sums its synapses and the resting charge, the veto dendrite its synapses alone.
        layer.veto.fill_(1.9)
    veto, _ = layer.accumulator_bounds()
    assert veto > own and veto == 64 * 486
    limit = int(veto) // 2
    assert layer.scale_synapses(limit) >= 64
    # Rounding moves each of a dendrite's 64 synapses by half a tick at most.
    assert layer.accumulator_bounds()[0] <= limit


def test_a_veto_dendrite_adds_no_pulse():
    plain, vetoed, tokens = _stack(granule_veto=0, **_FINE), _stack(**_FINE), _bytes()
    vetoed.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        for layer in vetoed.granule_layers.values():
            layer.veto.copy_(layer.receive * 8)
    totals = []
    for model in (plain, vetoed):
        fired = []
        for module in model.modules():
            if isinstance(module, _IntegrateFire):
                module.register_forward_hook(
                    lambda population, _, counts, fired=fired: fired.append((population, emitted_pulses(population, counts))))
        model(tokens)
        totals.append(fired)
    # The veto dendrites are populations of their own that count nothing.
    dendrites = [count for population, count in totals[1] if population.dendrite]
    assert len(totals[1]) == len(totals[0]) + len(_LEVELS) == 2 + len(_LEVELS) * 9 + 1
    assert len(dendrites) == len(_LEVELS) and not any(dendrites)
    # Closed cells send fewer spikes, so the stack fires less with the vetoes than without.
    assert sum(count for _, count in totals[1]) < sum(count for _, count in totals[0])


@pytest.mark.parametrize("setting", [dict(granule_veto=3), dict(granule_veto=32), dict(granule_veto=4.0),
                                     dict(granule_veto=-4), dict(granule_gate=4), dict(granule_slope=False),
                                     dict(granule_budget=8, granule_slope=False)])
def test_a_layer_refuses_a_veto_it_cannot_have(setting):
    with pytest.raises(ValueError, match="veto|gated"):
        _stack(**setting)


def test_the_trainer_hands_the_veto_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --granule-veto 4 --pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128
    --ctx 16 --bs 2 --steps 2 --warmup 1 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1
    --final-eval-batches 1 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    layer = model.granule_layers["0"]
    assert model.cfg.granule_veto == 4 and layer.veto.shape == (1, 16, 32)
    assert type(layer.fire) is _VetoedFire and layer.fire.steps == 4
