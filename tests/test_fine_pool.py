"""A pool that fires often and adds a small quantum of threshold with every spike divides by the stream's activity."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _IntegrateFire, _ShuntedFire, _stream_pool, emitted_pulses

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True)
# Sixty-four quanta to a threshold with four at rest, a pool spike for every eight pulses of 64 channels, and a pool
# at the output.
_FINE = dict(pool_fine=16, pool_rest=4, pool_rate=32, output_pool=128)


def _stack(seed=0, grown=True, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    if grown:
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


def _stream(model, tokens):
    return model.input_emit(model.input(tokens) * model.cfg.resolution)


def _quanta(count, dtype=torch.float32):
    return torch.full((1,), count, dtype=dtype)


def test_a_stack_keeps_the_pool_of_a_block_unless_told_otherwise():
    assert (EventLMConfig.pool_fine, EventLMConfig.pool_rest, EventLMConfig.pool_rate) == (1, None, 1)
    assert EventLMConfig.output_pool == 0
    plain, tokens = _stack(), _bytes()
    for part in (plain.state_reads["0"], plain.granule_layers["0"]):
        # A pool spike for every 256 pulses of 64 channels, four quanta to a threshold and four at rest.
        assert (part.pool.threshold, part.pool.cap, part.rest_quanta, part.fire.quanta_per_threshold) == (256, 128, 4, 4)
    assert plain.output_pool is None and type(plain.output_emit) is _IntegrateFire
    assert torch.equal(plain(tokens), _stack(pool_fine=1, pool_rest=4, pool_rate=1)(tokens))
    # The settings add no synapse and move none.
    fine = _stack(**_FINE)
    assert set(plain.state_dict()) == set(fine.state_dict())
    assert all(torch.equal(value, fine.state_dict()[name]) for name, value in plain.state_dict().items())
    assert not torch.equal(plain(tokens), fine(tokens))


def test_a_fine_pool_fires_for_every_eighth_of_the_channels_in_pulses():
    cfg = EventLMConfig(**dict(_STACK, **_FINE))
    assert cfg.pool_quanta == 64 and cfg.pool_rest_quanta == 4
    assert EventLMConfig(**_STACK).pool_quanta == EventLMConfig(**_STACK).pool_rest_quanta == 4
    pool = _stream_pool(cfg)
    # Thirty-two spikes where a block's pool fires one, up to as many quanta as that pool can add.
    assert pool.threshold == 8 == cfg.dim // 8 and pool.cap == 128 * 16
    assert pool(torch.tensor([0.0, 3.0, 4.0, 8.0, 100.0, 1e6])).tolist() == [0, 0, 1, 1, 13, 2048]
    model = _stack(**_FINE)
    for part in (model.state_reads["0"], model.granule_layers["1"], model):
        other = part.output_pool if part is model else part.pool
        assert (other.threshold, other.cap) == (8, 2048)
    assert model.granule_layers["0"].rest_quanta == model.state_reads["1"].rest_quanta == 4


def test_a_neuron_under_a_fine_pool_counts_its_threshold_in_sixty_fourths():
    fire = _IntegrateFire(128, 8, False, quanta_per_threshold=64, capped=True)
    charge = torch.tensor([512.0, -512.0, 600.0, 100.0, 7.0, 8.0])
    # Sixty-four quanta are the threshold of 256 ticks.
    assert fire(charge, _quanta(64.0)).tolist() == [2, -2, 2, 0, 0, 0]
    # Thirty-two more raise it by half, to 384 ticks.
    assert fire(charge, _quanta(96.0)).tolist() == [1, -1, 2, 0, 0, 0]
    # With the pool silent four quanta stand: 16 ticks.
    assert fire(charge, _quanta(4.0)).tolist() == [32, -32, 38, 6, 0, 1]
    # The membrane holds 128 thresholds of charge, which a low threshold would turn into more spikes than the cap.
    large = torch.tensor([1e7, -1e7])
    assert fire(large, _quanta(4.0)).tolist() == [128, -128]
    assert _IntegrateFire(128, 8, False, quanta_per_threshold=64)(large, _quanta(4.0)).tolist() == [2048, -2048]
    # A threshold above a whole one lets fewer spikes through, as under a block's pool.
    assert fire(large, _quanta(128.0)).tolist() == [64, -64]
    # Whole numbers in, whole numbers out.
    whole = fire(charge.to(torch.int32), _quanta(96, torch.int32))
    assert whole.dtype == torch.int32 and whole.tolist() == [1, -1, 2, 0, 0, 0]


def test_the_slope_of_a_count_passes_through_the_pool():
    fire = _IntegrateFire(128, 8, False, quanta_per_threshold=64, capped=True)
    charge = torch.tensor([512.0, -300.0], requires_grad=True)
    quanta = _quanta(96.0).requires_grad_()
    fire(charge, quanta).sum().backward()
    torch.testing.assert_close(charge.grad, torch.full((2,), 1 / 384))
    torch.testing.assert_close(quanta.grad, torch.tensor([-(512 - 300) / (96 * 384)]))
    # A pool's own count has the slope of its charge over its threshold.
    pool = _stream_pool(EventLMConfig(**dict(_STACK, **_FINE)))
    activity = torch.tensor([100.0, 737.0], requires_grad=True)
    pool(activity).sum().backward()
    torch.testing.assert_close(activity.grad, torch.full((2,), 1 / 8))


def test_output_neurons_under_a_pool_fire_for_the_charge_over_their_threshold():
    fire, plain = _ShuntedFire(2048, 3, False, quanta_per_threshold=128), _IntegrateFire(2048, 3, False)
    charge = torch.tensor([0.0, 3.0, 4.0, -4.0, 8.0, 12.0, -20.0, 1e7, -1e7])
    # Without quanta, and with a whole threshold of them, they fire as plain neurons do.
    assert torch.equal(fire(charge), plain(charge))
    assert fire(charge, _quanta(128.0)).tolist() == plain(charge).tolist() == [0, 0, 1, -1, 1, 2, -3, 2048, -2048]
    # Twice the quanta are twice the threshold, and the largest count stays the cap.
    assert fire(charge, _quanta(256.0)).tolist() == [0, 0, 0, 0, 1, 1, -1, 2048, -2048]
    assert fire(torch.tensor([1e6]), _quanta(2049.0)).tolist() == [2048]
    # A quantum is a sixteenth of a tick of charge: one quantum of threshold fires sixteen spikes a tick.
    assert fire(torch.tensor([1.0, -1.0, 100.0]), _quanta(1.0)).tolist() == [16, -16, 1600]
    # Whole numbers in, whole numbers out.
    whole = fire(charge.to(torch.int32), _quanta(256, torch.int32))
    assert whole.dtype == torch.int32 and whole.tolist() == [0, 0, 0, 0, 1, 1, -1, 2048, -2048]
    # The slope is that of the charge over the threshold, and none at the cap.
    charge = torch.tensor([64.0, -24.0, 1e7], requires_grad=True)
    quanta = _quanta(256.0).requires_grad_()
    fire(charge, quanta).sum().backward()
    torch.testing.assert_close(charge.grad, torch.tensor([1 / 16, 1 / 16, 0.0]))
    torch.testing.assert_close(quanta.grad, torch.tensor([-(64 - 24) / (16 * 256)]))


def test_fine_pools_hold_the_firing_level_when_the_stream_grows():
    tokens = _bytes()
    growth = {}
    for name, settings in (("block", {}), ("fine", _FINE)):
        model = _stack(**settings)
        stream = _stream(model, tokens)
        layer, read = model.granule_layers["0"], model.state_reads["0"]
        cells = [layer.cells(scale * stream).abs().mean() for scale in (1, 3)]
        probes = [read.packets(scale * stream)[0].abs().mean() for scale in (1, 3)]
        growth[name] = (cells[1] / cells[0]).item(), (probes[1] / probes[0]).item()
    # Three times the pulses: under a block's pool the cells and the probes fire half as much again or more,
    # under a fine pool as before.
    assert min(growth["block"]) > 1.5
    assert all(0.95 < ratio < 1.1 for ratio in growth["fine"])


def test_the_output_pool_keeps_the_evidence_when_the_stream_grows():
    model, tokens = _stack(**_FINE), _bytes()
    with torch.no_grad():
        model.output.weight.normal_(0, 0.5, generator=torch.Generator().manual_seed(5))
    stream = _stream(model, tokens)

    def counts(spikes):
        return model.output_emit(model.output(spikes), 1 + model.output_pool(spikes.abs().sum(-1, keepdim=True)))

    once, thrice = counts(stream), counts(3 * stream)
    assert torch.equal(once, once.round()) and once.abs().max() == model.output_emit.cap
    # The charge triples and so does the threshold, up to the one quantum that stands at rest.
    assert torch.equal(model.output(3 * stream), 3 * model.output(stream))
    assert (thrice - once).abs().max() <= 0.02 * model.output_emit.cap
    # What the decoder fires is that count.
    silent = _stack(grown=False, **_FINE)
    assert torch.equal(silent(tokens), silent.output_emit(
        silent.output(_stream(silent, tokens)),
        1 + silent.output_pool(_stream(silent, tokens).abs().sum(-1, keepdim=True))))


@pytest.mark.parametrize("turned", [False, True])
def test_a_stack_with_fine_pools_fires_whole_spikes_and_sees_no_later_byte(turned):
    model, tokens = _stack(read_turn=turned, **_FINE), _bytes()
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
    read = model.state_reads["0"]
    for packet in read.packets(_stream(model, tokens)):
        assert torch.equal(packet, packet.round()) and (packet > 0).any() and (packet < 0).any()
    if turned:
        assert read.turned.quanta_per_threshold == 64 and read.turned.capped
    cells = model.granule_layers["0"].cells(_stream(model, tokens))
    assert torch.equal(cells, cells.round()) and 0 < (cells > 0).float().mean() < 1


def test_every_synapse_of_a_stack_with_fine_pools_learns():
    model = _stack(**_FINE)
    model(_bytes()).float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_fine_pools_stay_within_the_range_of_exact_sums():
    bounds = {}
    for name, settings in (("block", {}), ("fine", _FINE)):
        model = _stack(**settings)
        model._check_accumulator()
        layer = model.granule_layers["1"]
        with torch.no_grad():
            layer.receive.fill_(1.9)
        bounds[name] = layer.accumulator_bounds()[0].item()
    # A cell sums its synapses and its resting charge at the strongest inhibition: sixteen spikes of rest, each a
    # quarter of four quanta of 16 ticks under a block's pool and a quarter of a quantum of one tick under a fine
    # one, at the most quanta, per unit of a stream neuron's 512 spikes.
    assert bounds["block"] == 64 * 486 + 16 * 4 * (4 + 128) * 16 // 512
    assert bounds["fine"] == 64 * 486 + -(-16 * 4 * (4 + 2048) // 512)


def test_the_pulses_of_fine_pools_are_counted():
    pulses = {}
    for name, settings in (("block", {}), ("fine", _FINE)):
        model, tokens, fired = _stack(**settings), _bytes(), {}
        for module_name, module in model.named_modules():
            if isinstance(module, _IntegrateFire):
                module.register_forward_hook(lambda population, _, counts, key=module_name: fired.setdefault(
                    key, []).append(emitted_pulses(population, counts).item()))
        model(tokens)
        pulses[name] = fired
    # The input and the output, and in every level the read's pool, its neurons and its write, and the cells' pool,
    # the cells and their write. A fine stack has the output's pool too.
    assert len(pulses["block"]) == 2 + len(_LEVELS) * 6 and set(pulses["fine"]) - set(pulses["block"]) == {"output_pool"}
    # The first pool hears the same stream in both: thirty-two times the spikes, up to the rounding of each count.
    block, fine = pulses["block"]["state_reads.0.pool"][0], pulses["fine"]["state_reads.0.pool"][0]
    assert abs(fine - 32 * block) <= 16 * tokens.numel()
    assert pulses["fine"]["output_pool"][0] > 0


@pytest.mark.parametrize("setting", [
    dict(pool_fine=3), dict(pool_fine=0), dict(pool_rate=12), dict(pool_rate=1024), dict(pool_rest=0),
    dict(pool_rest=2.0), dict(pool_fine=32), dict(pool_fine=16, granule_gate=4, granule_cells=64),
    dict(pool_fine=16, granule_budget=8, granule_slope=False),
    dict(pool_rate=2, read_mode="soft", read_budget=None),
])
def test_a_pool_refuses_settings_it_cannot_have(setting):
    with pytest.raises(ValueError, match="pool|quantum"):
        _stack(**setting)


@pytest.mark.parametrize("setting", [dict(output_pool=-1), dict(output_pool=1.5), dict(output_pool=128, stack=False)])
def test_only_a_stack_gives_its_output_neurons_a_pool(setting):
    settings = dict(_STACK, **setting)
    if not settings["stack"]:
        settings = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8,
                        output_pool=128)
    with pytest.raises(ValueError, match="output neurons of a stack"):
        EventLM(EventLMConfig(**settings))


def test_the_trainer_hands_the_pool_settings_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 1 --spike-resolution 16
    --router-charge-bits 8 --read-after 0 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0 --granule-clusters 1 --granule-fan 32 --granule-cells 16
    --granule-slope --pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128 --ctx 16 --bs 2 --steps 2
    --warmup 1 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1 --final-eval-batches 1
    --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    cfg = model.cfg
    assert (cfg.pool_fine, cfg.pool_rest, cfg.pool_rate, cfg.output_pool) == (16, 4, 32, 128)
    layer = model.granule_layers["0"]
    assert (layer.pool.threshold, layer.rest_quanta, layer.fire.quanta_per_threshold) == (4, 4, 64)
    assert type(model.output_emit) is _ShuntedFire and model.output_pool.threshold == 4
