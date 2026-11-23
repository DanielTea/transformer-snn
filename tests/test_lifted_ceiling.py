"""A pool that divides a neuron's charge before the membrane bounds it leaves the neuron its largest count."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _ExcitatoryFire, _IntegrateFire

_LEVELS = (0, 1)
# Sixty-four quanta to a threshold with four at rest, a pool spike for every eight pulses of 64 channels, a pool at
# the output, and neurons that fire 128 fine spikes at most.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, read_cap=128, granule_after=_LEVELS, granule_clusters=1,
              granule_fan=64, granule_cells=64, granule_slope=True, granule_cap=128, pool_fine=16, pool_rest=4,
              pool_rate=32, output_pool=128, pool_lifts_ceiling=True)


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=80):
    return torch.randint(0, 256, (2, length), generator=torch.Generator().manual_seed(seed))


def _stream(model, tokens, loud=1):
    return model.input_emit(model.input(tokens) * 16) * loud


def _fire(lifted, kind=_IntegrateFire):
    # A whole threshold of 256 ticks counts 64 quanta, and the membrane holds 128 whole thresholds.
    fire = kind(128, 8, False, quanta_per_threshold=64, capped=True)
    fire.lifted = lifted
    return fire


def test_a_lifted_neuron_fires_up_to_its_cap_however_high_the_threshold_stands():
    # Under 512 quanta a fine spike takes 2048 ticks, and the membrane of 128 whole thresholds holds 16 of them.
    quanta, spike = torch.tensor([512.0]), 2048
    thresholds = torch.tensor([-200, -100, -20, -16, 0, 0.4, 0.5, 16, 17, 100, 128, 200])
    charge = (thresholds * spike).round()
    assert _fire(False)(charge, quanta).tolist() == [-16, -16, -16, -16, 0, 0, 1, 16, 16, 16, 16, 16]
    assert _fire(True)(charge, quanta).tolist() == [-128, -100, -20, -16, 0, 0, 1, 16, 17, 100, 128, 128]
    # A cell has one rail.
    assert _fire(True, _ExcitatoryFire)(charge, quanta).tolist() == [0, 0, 0, 0, 0, 0, 1, 16, 17, 100, 128, 128]
    # Whole numbers of charge fire the same counts.
    assert torch.equal(_fire(True)(charge.to(torch.int32), quanta.to(torch.int32)),
                       _fire(True)(charge, quanta).to(torch.int32))


@pytest.mark.parametrize("quanta", [4, 16, 64, 100, 512, 2052])
def test_a_lifted_neuron_fires_as_before_below_its_old_ceiling(quanta):
    # The old ceiling is the membrane's 8192 quanta of charge, and 128 spikes under a weak pool.
    ceiling = min(8192 // quanta, 128) * quanta * 4
    generator = torch.Generator().manual_seed(quanta)
    charge = torch.randint(-ceiling, ceiling + 1, (4096,), generator=generator).float()
    pool = torch.tensor([float(quanta)])
    assert torch.equal(_fire(True)(charge, pool), _fire(False)(charge, pool))
    assert torch.equal(_fire(True, _ExcitatoryFire)(charge, pool), _fire(False, _ExcitatoryFire)(charge, pool))


def test_a_lifted_neuron_learns_up_to_its_ceiling():
    quanta, spike = torch.tensor([512.0]), 2048
    for lifted, slopes in ((False, [1, 0, 0]), (True, [1, 1, 0])):
        charge = (torch.tensor([10.0, 100.0, 200.0]) * spike).requires_grad_()
        _fire(lifted)(charge, quanta).sum().backward()
        # A spike more for every threshold of charge, below the ceiling.
        assert torch.allclose(charge.grad * spike, torch.tensor(slopes, dtype=torch.float))
    # The pool learns that a higher threshold costs a neuron below its ceiling spikes, and one above it none.
    pool = quanta.clone().requires_grad_()
    _fire(True)(torch.tensor([100.0 * spike]), pool).sum().backward()
    assert torch.allclose(pool.grad, torch.tensor([-100 / 512]))
    pool = quanta.clone().requires_grad_()
    _fire(True)(torch.tensor([200.0 * spike]), pool).sum().backward()
    assert pool.grad.item() == 0


def test_only_the_neurons_of_reads_and_plain_cells_are_lifted():
    lifted, plain = _stack(), _stack(pool_lifts_ceiling=False)
    # The ceiling is no synapse: both stacks have the same ones.
    assert all(torch.equal(first, second) for first, second in zip(lifted.state_dict().values(),
                                                                    plain.state_dict().values()))
    for model, state in ((lifted, True), (plain, False)):
        populations = {name for name, module in model.named_modules() if getattr(module, "lifted", False)}
        expected = {f"{part}.{level}.fire" for part in ("state_reads", "granule_layers") for level in _LEVELS}
        assert populations == (expected if state else set())


def test_reads_pass_their_old_ceiling_under_a_loud_stream():
    lifted, plain, tokens = _stack(), _stack(pool_lifts_ceiling=False), _bytes()
    for loud, passes in ((1, False), (8, True)):
        spikes = _stream(lifted, tokens, loud)
        read, other = lifted.state_reads["0"], plain.state_reads["0"]
        quanta = read.rest_quanta + read.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
        # The old ceiling of a neuron at each byte, as [batch, heads, cells, neurons].
        old = (8192 // quanta).clamp(max=128)[:, None]
        for packet, bounded in zip(read.packets(spikes), other.packets(spikes)):
            assert torch.equal(packet, packet.round()) and packet.abs().max() <= 128
            assert (bounded.abs() <= old).all()
            # Below its old ceiling a neuron fires what it fired, and at the ceiling it now fires on.
            below = bounded.abs() < old
            assert torch.equal(packet[below], bounded[below]) and (packet.abs() >= bounded.abs()).all()
            assert (packet.abs() > old).any() == passes


def test_cells_pass_their_old_ceiling_under_a_loud_stream():
    lifted, plain, tokens = _stack(), _stack(pool_lifts_ceiling=False), _bytes()
    with torch.no_grad():
        for model in (lifted, plain):
            # Cells that collect a lot of charge.
            model.granule_layers["0"].receive.mul_(8)
    spikes = _stream(lifted, tokens, 8)
    layer, other = lifted.granule_layers["0"], plain.granule_layers["0"]
    quanta = layer.rest_quanta + layer.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
    old = (8192 // quanta).clamp(max=128)[..., None]
    fired, bounded = layer.cells(spikes, tokens), other.cells(spikes, tokens)
    assert torch.equal(fired, fired.round()) and 0 <= fired.min() and fired.max() <= 128
    assert (bounded <= old).all() and (fired > old).any()
    below = bounded < old
    assert torch.equal(fired[below], bounded[below]) and (fired >= bounded).all()


def test_a_stack_with_lifted_ceilings_fires_whole_spikes_and_sees_no_later_byte():
    model, tokens = _stack(), _bytes()
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
    # The largest count of a neuron is what it was, so the exact sums are as they were.
    model._check_accumulator()
    plain = _stack(pool_lifts_ceiling=False)
    for name in ("state_reads", "granule_layers"):
        for level in map(str, _LEVELS):
            bounds, others = (getattr(stack, name)[level].accumulator_bounds() for stack in (model, plain))
            assert all(torch.equal(torch.as_tensor(bound), torch.as_tensor(other))
                       for bound, other in zip(bounds, others))


def test_every_synapse_of_a_stack_with_lifted_ceilings_learns():
    model = _stack()
    model(_bytes()).float().square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("state_reads", "granule_layers")) and parameter.ndim >= 2:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("setting", [dict(pool_lifts_ceiling=1), dict(read_after=None, granule_after=None),
                                     dict(read_mode="soft", read_budget=None, read_cap=None),
                                     dict(granule_veto=4),
                                     dict(granule_budget=8, granule_slope=False)])
def test_a_stack_refuses_ceilings_that_no_pool_lifts(setting):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **setting)))


def test_the_trainer_hands_the_lifted_ceiling_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --read-cap 128 --granule-after 0 --granule-clusters 1 --granule-fan 32
    --granule-cells 16 --granule-slope --granule-cap 128 --pool-fine 16 --pool-rest 4 --pool-rate 32
    --output-pool 128 --pool-lifts-ceiling
    --ctx 16 --bs 2 --steps 2 --warmup 1 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1
    --final-eval-batches 1 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    assert model.cfg.pool_lifts_ceiling
    assert model.state_reads["0"].fire.lifted and model.granule_layers["0"].fire.lifted
