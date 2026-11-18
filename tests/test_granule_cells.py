import pytest
import torch
from torch.nn import functional as F

from snnformer.event_lm import (EventLM, EventLMConfig, _beat_views, _expected_beat_views, _IntegrateFire,
                                _spread_channels, _weight_ticks, emitted_pulses)


def _decoder(seed=0, **settings):
    base = dict(dim=96, hidden=40, depth=6, route="steered_correction_code", carry=False, memory_heads=4,
                memory_slots=14, resolution=16, memory_width=32, forecast_key_channels=2, match_forecast_key=True,
                learn_context_keys=True, compact_delays=True, compact_axon_state=True, activity_inhibition=True,
                inhibition_by_expansion=True, local_trace=True, trace_write_credit=True, router_charge_bits=8,
                residual_correction_start=3, compact_trace_state=True, delay_group_widths=(12, 12, 8, 8),
                trace_arrival="selected", trace_arrival_groups=2, clock_start=4, clock_threshold=8)
    torch.manual_seed(seed)
    return EventLM(EventLMConfig(**dict(base, **settings)))


def _granules(**settings):
    return _decoder(**dict(dict(granule_after=(0, 2), granule_clusters=6, granule_cells=20, granule_fan=24), **settings))


def _grow(model):
    """Give the silent layers output synapses."""
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _stream(seed=2):
    return torch.randint(-20, 21, (2, 40, 96), generator=torch.Generator().manual_seed(seed)).float()


def test_untrained_granule_layers_leave_the_decoder_as_it_is():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    plain, model = _decoder(), _granules()
    # The layers are created last and their output synapses start at zero.
    assert torch.equal(model(tokens), plain(tokens))
    assert not torch.equal(_grow(model)(tokens), plain(tokens))
    # With state reads the layers still come last.
    reads = dict(read_after=(1, 3), read_width=32, read_window=24)
    assert torch.equal(_granules(**reads)(tokens), _decoder(**reads)(tokens))


def test_cells_fire_whole_excitatory_spikes_above_their_rest():
    layer = _granules(granule_gain=1.0, granule_rest=0.0).granule_layers["0"]
    spikes = _stream()
    cells = layer.cells(spikes)
    assert cells.shape == (2, 40, 6, 20)
    assert torch.equal(cells, cells.round()) and cells.min() == 0 and cells.max() > 1
    # A cell whose inputs all push it down stays silent, and a silent stream fires nothing.
    with torch.no_grad():
        down = layer.cells(spikes) * 0
        layer.receive.copy_(-layer.receive.abs())
        assert torch.equal(layer.cells(spikes.abs()), down)
    assert layer.cells(torch.zeros_like(spikes)).abs().sum() == 0
    # More resting charge below the threshold leaves fewer spikes, quarter spike by quarter spike.
    counts = []
    for rest in (0.0, 0.25, 1.0, 4.0):
        resting = _granules(granule_gain=1.0, granule_rest=rest).granule_layers["0"]
        assert torch.equal(resting.rest_quarters(), torch.full_like(resting.rest, 4 * rest))
        counts.append(resting.cells(spikes))
    assert all((later <= earlier).all() for earlier, later in zip(counts, counts[1:], strict=False))
    assert counts[0].sum() > counts[1].sum() > counts[2].sum() > counts[3].sum()
    # One spike of rest removes one spike from every cell that fired more than one.
    assert torch.equal(counts[2][counts[0] > 1], counts[0][counts[0] > 1] - 1)


def test_clusters_listen_to_distinct_channels_that_cover_the_stream_evenly():
    generator = torch.Generator().manual_seed(5)
    channels = _spread_channels(96, 10, 24, generator)
    assert channels.shape == (10, 24)
    assert all(len(set(row.tolist())) == 24 for row in channels)
    use = torch.bincount(channels.flatten(), minlength=96)
    assert use.max() - use.min() <= 1
    first, second = (_granules().granule_layers[index] for index in ("0", "2"))
    again = _granules(seed=9).granule_layers["0"]
    # The wiring depends on the layer's place and not on the weights' seed, and it is saved with the weights.
    assert torch.equal(first.inputs, again.inputs) and not torch.equal(first.inputs, second.inputs)
    assert not torch.equal(first.inputs, first.outputs)
    assert {"granule_layers.0.inputs", "granule_layers.0.outputs"} <= set(_granules().state_dict())


def test_cells_only_change_the_channels_their_axons_reach():
    layer = _grow(_granules(granule_gain=1.0, granule_rest=0.0)).granule_layers["0"]
    spikes = _stream()
    with torch.no_grad():
        layer.send[1:] = 0
    changed = (layer(spikes) != spikes).any(0).any(0)
    reached = torch.zeros(96, dtype=torch.bool)
    reached[layer.outputs[0]] = True
    assert changed.any() and not (changed & ~reached).any()
    assert torch.equal(layer(spikes), layer(spikes).round())
    # A cell hears nothing from the channels outside its cluster's fan. Swapping those keeps the pool's inhibition.
    outside = torch.ones(96, dtype=torch.bool)
    outside[layer.inputs[0]] = False
    other = spikes.clone()
    other[..., outside] = spikes[..., outside].flip(-1)
    assert not torch.equal(other, spikes)
    assert torch.equal(layer.cells(other)[..., 0, :], layer.cells(spikes)[..., 0, :])
    assert not torch.equal(layer.cells(other)[..., 1:, :], layer.cells(spikes)[..., 1:, :])


@pytest.mark.parametrize("budget, delays, relays, traces, beats, clock, hold", [
    (None, 1, 0, 0, 0, "word", "mean"), (6, 1, 0, 0, 0, "word", "mean"), (None, 3, 0, 0, 0, "word", "mean"),
    (6, 4, 0, 0, 0, "word", "mean"), (None, 1, 30, 0, 0, "word", "mean"), (6, 4, 24, 0, 0, "word", "mean"),
    (6, 2, 0, 2, 0, "word", "mean"), (None, 1, 0, 1, 0, "word", "mean"), (6, 3, 24, 3, 0, "word", "mean"),
    (6, 1, 0, 0, 3, "word", "mean"), (None, 2, 0, 0, 2, "word", "last"), (6, 1, 24, 0, 3, "every", "sum"),
    (6, 2, 0, 1, 2, "learned", "mean"), (None, 1, 0, 0, 4, "learned", "sum"), (6, 4, 24, 0, 2, "learned", "last"),
    (6, 2, 0, 0, 3, "end", "sum")])
def test_granule_layers_keep_the_decoder_causal_and_train(budget, delays, relays, traces, beats, clock, hold):
    model = _grow(_granules(granule_gain=1.0, granule_budget=budget, granule_delays=delays, granule_relays=relays,
                            granule_traces=traces, granule_beats=beats, granule_clock=clock, granule_hold=hold,
                            read_after=(1, 3), read_width=32, read_window=24)).train()
    tokens = torch.randint(0, 256, (2, 64), generator=torch.Generator().manual_seed(1))
    later = tokens.clone()
    later[:, 40:] = (later[:, 40:] + 7) % 256
    with torch.no_grad():
        first, second = model(tokens), model(later)
    assert torch.equal(first[:, :40], second[:, :40]) and not torch.equal(first[:, 40:], second[:, 40:])
    F.cross_entropy(model(tokens).flatten(0, 1) / 16, tokens.flatten()).backward()
    for name, parameter in model.granule_layers.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_cell_spikes_are_counted_and_the_integer_guard_sees_the_layers():
    model = _grow(_granules(granule_gain=1.0, granule_rest=0.0))
    layer = model.granule_layers["0"]
    counted = []
    for module in layer.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda _module, _inputs, counts: counted.append(counts.abs().sum()))
    spikes = _stream()
    layer(spikes)
    cells = layer.cells(spikes).sum()
    # The pool, the cells and the stream's firing population each report their pulses once per pass.
    assert len(counted) == 5 and cells > 0 and cells in counted
    model._check_accumulator()
    # Eighty cells at the largest weight would overflow the exact sum of one stream channel.
    crowded = _granules(granule_cells=80)
    crowded._check_accumulator()
    with torch.no_grad():
        crowded.granule_layers["2"].send.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        crowded._check_accumulator()


def test_the_budget_counts_the_cells_and_its_price_pushes_on_them_by_their_share():
    spikes = _stream()
    totals, gradients = [], []
    for share in (0.0, 0.25, 1.0):
        layer = _granules(granule_gain=1.0, granule_rest=0.0, granule_price=share).granule_layers["0"]
        pulses = []
        for module in layer.modules():
            if isinstance(module, _IntegrateFire):
                module.register_forward_hook(lambda module, _inputs, counts: pulses.append(emitted_pulses(module, counts)))
        layer(spikes)
        total = torch.stack(pulses).sum()
        totals.append(total.item())
        # The output synapses are silent, so a price can only reach the input synapses through the cells' own spikes.
        gradient, = torch.autograd.grad(total, layer.receive, allow_unused=True)
        gradients.append(torch.zeros_like(layer.receive) if gradient is None else gradient)
    assert totals[0] == totals[1] == totals[2] > 0
    assert gradients[0].abs().sum() == 0 and gradients[2].abs().sum() > 0
    assert torch.allclose(gradients[1], 0.25 * gradients[2])
    # Only the cells are spared. Every other population of the decoder takes the full price.
    spared = [name for name, module in _granules().named_modules()
              if isinstance(module, _IntegrateFire) and module.price_share != 1]
    assert spared == ["granule_layers.0.fire", "granule_layers.2.fire"]


def test_cells_can_keep_finer_synapses_than_the_blocks():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    assert torch.equal(_granules(granule_bits=11)(tokens), _decoder()(tokens))
    models = [_grow(_granules(granule_gain=1.0, granule_rest=0.0, granule_bits=bits)) for bits in (None, 11)]
    coarse, fine = (model.granule_layers["0"] for model in models)
    assert (coarse.bits, fine.bits) == (8, 11) and fine.fire.threshold == fine.emit.threshold == 1 << 11
    # The same weights land on a grid that is eight times finer, so fewer output synapses round to nothing.
    ticks = [_weight_ticks(layer.send, charge_bits=layer.bits) for layer in (coarse, fine)]
    assert torch.equal(ticks[1], (fine.send * 2048).round())
    assert (ticks[1] == 0).float().mean() < 0.5 * (ticks[0] == 0).float().mean()
    # The layers still do nearly the same thing, in whole spikes, and the integer guard covers the finer sums.
    spikes = _stream()
    assert (fine.cells(spikes) - coarse.cells(spikes)).abs().mean() < 0.05 * coarse.cells(spikes).mean()
    assert torch.equal(fine(spikes), fine(spikes).round())
    assert (fine(spikes) - coarse(spikes)).abs().mean() < 0.5 * (coarse(spikes) - spikes).abs().mean()
    models[1]._check_accumulator()
    for bits in (5, 16, 8.0):
        with pytest.raises(ValueError, match="charge bits"):
            _granules(granule_bits=bits)


def test_racing_cells_spend_their_budget_whatever_their_charges_are():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    assert torch.equal(_granules(granule_budget=6)(tokens), _decoder()(tokens))
    layer = _granules(granule_gain=1.0, granule_budget=6).granule_layers["0"]
    counted = []
    layer.fire.register_forward_hook(lambda module, _inputs, counts: counted.append(emitted_pulses(module, counts)))
    spikes = _stream()
    cells = layer.cells(spikes)
    assert cells.shape == (2, 40, 6, 20) and torch.equal(cells, cells.round()) and cells.min() == 0
    assert counted[0] == cells.sum()
    spent = cells.sum(-1)
    assert spent.min() >= 4 and spent.max() <= 8 and abs(spent.mean().item() - 6) < 0.1
    # The spikes of a race do not depend on the size of its charges, so a price on pulses has nothing to push on.
    gradient, = torch.autograd.grad(cells.sum(), layer.receive)
    assert gradient.abs().max() < 1e-3
    busiest = cells[..., 0, :].sum((0, 1)).argmax()
    with torch.no_grad():
        saved = layer.receive.clone()
        layer.receive.mul_(0.05)
        assert abs(layer.cells(spikes).sum(-1).mean().item() - 6) < 0.1
        layer.receive.copy_(saved)
        # A sharper cluster gives its spikes to fewer cells, and the most charged cell stays ahead.
        layer.sharpness.fill_(2.0)
        sharp = layer.cells(spikes)
        layer.sharpness.zero_()
        # A cell's rest is its handicap in the race.
        layer.rest[0, busiest] = 8.0
        handicapped = layer.cells(spikes)
    assert (sharp > 0).sum() < (cells > 0).sum()
    assert (sharp.argmax(-1) == cells.argmax(-1)).float().mean() > 0.9
    assert handicapped[..., 0, busiest].sum() < 0.5 * cells[..., 0, busiest].sum()
    assert abs(handicapped.sum(-1).mean().item() - 6) < 0.1
    for budget in (0, 513, 6.0):
        with pytest.raises(ValueError, match="whole budget"):
            _granules(granule_budget=budget)


def test_the_cells_of_a_cluster_can_race_in_equal_groups():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    assert torch.equal(_granules(granule_budget=3, granule_races=4)(tokens), _decoder()(tokens))
    model = _grow(_granules(granule_gain=1.0, granule_budget=3, granule_races=4))
    layer = model.granule_layers["0"]
    spikes = _stream()
    cells = layer.cells(spikes)
    assert cells.shape == (2, 40, 6, 20) and torch.equal(cells, cells.round()) and cells.min() == 0
    # Each group of five cells spends the budget of three spikes.
    spent = cells.unflatten(-1, (4, 5)).sum(-1)
    assert spent.min() >= 1 and spent.max() <= 5 and abs(spent.mean().item() - 3) < 0.1
    # A cell races only against the cells of its own group.
    with torch.no_grad():
        busiest = 5 + cells[..., 0, 5:10].sum((0, 1)).argmax()
        layer.rest[0, busiest] = 8.0
        handicapped = layer.cells(spikes)
    assert handicapped[..., 0, busiest].sum() < 0.5 * cells[..., 0, busiest].sum()
    assert not torch.equal(handicapped[..., 0, 5:10], cells[..., 0, 5:10])
    assert torch.equal(handicapped[..., 0, :5], cells[..., 0, :5])
    assert torch.equal(handicapped[..., 0, 10:], cells[..., 0, 10:]) and torch.equal(handicapped[..., 1:, :], cells[..., 1:, :])
    # The decoder with grouped races stays causal and trains.
    later = tokens.clone()
    later[:, 40:] = (later[:, 40:] + 7) % 256
    with torch.no_grad():
        first, second = model(tokens), model(later)
    assert torch.equal(first[:, :40], second[:, :40]) and not torch.equal(first[:, 40:], second[:, 40:])
    F.cross_entropy(model.train()(tokens).flatten(0, 1) / 16, tokens.flatten()).backward()
    assert all(torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0
               for name, parameter in layer.named_parameters() if name in ("receive", "send"))
    for settings in ({"granule_races": 0}, {"granule_races": 3}, {"granule_races": 2.0}, {"granule_budget": None}):
        with pytest.raises(ValueError, match="equal groups"):
            _granules(**dict({"granule_budget": 3, "granule_races": 4}, **settings))


def test_cells_learn_from_a_softer_race_and_fire_in_the_race_as_it_is():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    sharp = _grow(_granules(granule_gain=1.0, granule_budget=3))
    soft = _grow(_granules(granule_gain=1.0, granule_budget=3, granule_soften=4.0))
    soft.load_state_dict(sharp.state_dict())
    # One cell of every cluster is twelve thresholds behind the others, which are level.
    slopes = []
    for model in (sharp, soft):
        fire = model.granule_layers["0"].fire
        charges = torch.zeros(1, 1, 6, 20)
        charges[..., 0] = -12.0
        charges.requires_grad_()
        cells = fire(charges * (fire.threshold // fire.quanta_per_threshold), torch.ones(1, 1, 6, 1))
        cells[..., 0].sum().backward()
        slopes.append((cells.detach(), charges.grad[..., 0].sum().item()))
    # The spikes are the same: about three per cluster, none from the cell behind. That cell learns from the
    # softer race.
    assert torch.equal(slopes[0][0], slopes[1][0]) and torch.equal(slopes[0][0], slopes[0][0].round())
    spent = slopes[0][0].sum(-1)
    assert spent.min() >= 2 and spent.max() <= 4 and slopes[0][0][..., 0].sum() == 0
    assert 0 < slopes[0][1] < 1e-4 and slopes[1][1] > 1e-2
    with torch.no_grad():
        assert torch.equal(sharp(tokens), soft(tokens))
    F.cross_entropy(soft.train()(tokens).flatten(0, 1) / 16, tokens.flatten()).backward()
    assert all(torch.isfinite(parameter.grad).all() and parameter.grad.abs().max() > 0
               for name, parameter in soft.granule_layers["0"].named_parameters() if name in ("receive", "send"))
    for settings in ({"granule_soften": 0.5}, {"granule_soften": float("inf")}, {"granule_budget": None}):
        with pytest.raises(ValueError, match="softened"):
            _granules(**dict({"granule_budget": 3, "granule_soften": 2.0}, **settings))


def test_fibres_bring_their_channels_after_their_own_delays():
    tokens = torch.randint(0, 256, (2, 48), generator=torch.Generator().manual_seed(3))
    assert torch.equal(_granules(granule_delays=4)(tokens), _decoder()(tokens))
    model = _granules(granule_gain=1.0, granule_rest=0.0, granule_delays=4)
    layer, present = model.granule_layers["0"], _granules(granule_gain=1.0, granule_rest=0.0).granule_layers["0"]
    # The wiring of channels is the one a layer without delays has, and each delay takes a quarter of a cluster's fibres.
    assert torch.equal(layer.inputs, present.inputs) and torch.equal(layer.outputs, present.outputs)
    assert "granule_layers.0.taps" in model.state_dict() and not hasattr(present, "taps")
    assert all(torch.equal((layer.taps == delay).sum(-1), torch.full((6,), 6)) for delay in range(4))
    assert not torch.equal(layer.taps, model.granule_layers["2"].taps)
    spikes = _stream()
    heard = layer.heard(spikes)
    assert torch.equal(present.heard(spikes), spikes[..., present.inputs])
    for delay in range(4):
        earlier = F.pad(spikes, (0, 0, delay, 0))[:, :spikes.shape[1]][..., layer.inputs]
        assert torch.equal(heard[..., layer.taps == delay], earlier[..., layer.taps == delay])
    # One byte of activity charges the cells on that byte and on the three after it, and on none before.
    pulse = torch.zeros_like(spikes)
    pulse[:, 10] = spikes[:, 10]
    fired = layer.cells(pulse).abs().sum((0, 2, 3))
    assert torch.equal(fired.nonzero().flatten(), torch.arange(10, 14))
    assert torch.equal(present.cells(pulse).abs().sum((0, 2, 3)).nonzero().flatten(), torch.tensor([10]))
    for delays in (0, 25, 2.0):
        with pytest.raises(ValueError, match="one delay"):
            _granules(granule_delays=delays)


def test_slow_fibres_bring_what_the_last_bytes_left_behind():
    tokens = torch.randint(0, 256, (2, 48), generator=torch.Generator().manual_seed(3))
    assert torch.equal(_granules(granule_delays=2, granule_traces=2)(tokens), _decoder()(tokens))
    model = _granules(granule_gain=1.0, granule_rest=0.0, granule_delays=2, granule_traces=2)
    layer, present = model.granule_layers["0"], _granules(granule_gain=1.0, granule_rest=0.0).granule_layers["0"]
    # Two delays and two kinds of slow synapses each take a quarter of a cluster's fibres, on the same channels.
    assert torch.equal(layer.inputs, present.inputs) and torch.equal(layer.outputs, present.outputs)
    assert all(torch.equal((layer.taps == kind).sum(-1), torch.full((6,), 6)) for kind in range(4))
    spikes = _stream()
    passed = layer.held(spikes)
    assert len(passed) == 2
    for share, trace in zip((0.25, 0.0625), passed, strict=True):
        # A slow synapse adds what arrives to the charge it holds and passes its share on.
        charge = torch.zeros(2, 96, dtype=torch.float64)
        for byte in range(spikes.shape[1]):
            charge = charge * (1 - share) + spikes[:, byte].double()
            assert torch.allclose(trace[:, byte].double(), charge * share, atol=1e-4)
    heard = layer.heard(spikes)
    views = [spikes, F.pad(spikes, (0, 0, 1, 0))[:, :spikes.shape[1]], *passed]
    for kind, view in enumerate(views):
        assert torch.equal(heard[..., layer.taps == kind], view[..., layer.inputs][..., layer.taps == kind])
    # One byte of activity is heard on that byte and for many bytes after it, and on none before.
    pulse = torch.zeros_like(spikes)
    pulse[:, 10] = spikes[:, 10]
    echo = layer.heard(pulse).abs().sum((0, 2, 3))
    assert echo[:10].sum() == 0 and (echo[10:] > 0).all() and (echo[12:].diff() < 0).all()
    assert torch.equal(present.heard(pulse).abs().sum((0, 2, 3)).nonzero().flatten(), torch.tensor([10]))
    # The held charge stays exact over a stream that is longer than one stretch of the sum.
    long = torch.randint(-20, 21, (1, 1300, 8), generator=torch.Generator().manual_seed(4)).float()
    charge = torch.zeros(1, 8, dtype=torch.float64)
    for byte in range(1300):
        charge = charge * 0.75 + long[:, byte].double()
    slow = layer.held(long)[0]
    assert torch.isfinite(slow).all() and torch.allclose(slow[:, -1].double(), charge * 0.25, atol=1e-4)
    assert torch.equal(slow[:, :700], layer.held(long[:, :700])[0])
    for settings in (dict(granule_traces=5), dict(granule_traces=-1), dict(granule_traces=1.0),
                     dict(granule_delays=22, granule_traces=3)):
        with pytest.raises(ValueError, match="slow fibres"):
            _granules(**settings)


def test_cells_can_hear_relays_of_the_whole_stream():
    tokens = torch.randint(0, 256, (2, 48), generator=torch.Generator().manual_seed(3))
    assert torch.equal(_granules(granule_relays=30)(tokens), _decoder()(tokens))
    model = _grow(_granules(granule_gain=1.0, granule_rest=0.0, granule_relays=30, granule_delays=3))
    layer, direct = model.granule_layers["0"], _granules(granule_gain=1.0, granule_rest=0.0).granule_layers["0"]
    assert layer.relay.weight.shape == (30, 96) and direct.relay is None
    assert "granule_layers.0.relay.weight" in model.state_dict()
    # The fibres of a cluster come from distinct relays, and every relay has clusters that hear it.
    assert layer.inputs.shape == (6, 24) and all(len(set(row.tolist())) == 24 for row in layer.inputs)
    assert torch.bincount(layer.inputs.flatten(), minlength=30).min() >= 2
    # With as many relays as fibres every cluster hears every relay.
    assert torch.equal(_granules(granule_relays=24).granule_layers["0"].inputs, torch.arange(24).expand(6, 24))
    spikes = _stream()
    fibres = layer.fibres(spikes)
    assert fibres.shape == (2, 40, 30) and torch.equal(fibres, fibres.round()) and fibres.min() < 0 < fibres.max()
    # A relay fires one spike for every threshold of charge, on either rail.
    charge = F.linear(spikes, _weight_ticks(layer.relay.weight, charge_bits=8))
    assert torch.equal(fibres, ((charge.abs() + 128) // 256) * charge.sign())
    assert direct.fibres(spikes) is spikes
    # A stream channel now reaches the cells of every cluster, where a fibre from the stream reaches a quarter of them.
    other = spikes.clone()
    other[..., 5] += 16
    changed = (layer.cells(other) != layer.cells(spikes)).any(0).any(0).any(-1)
    assert changed.all()
    reached = (direct.inputs == 5).any(-1)
    assert torch.equal((direct.cells(other) != direct.cells(spikes)).any(0).any(0).any(-1), reached) and not reached.all()
    assert torch.equal(layer(spikes), layer(spikes).round())
    # The relays' spikes are counted, and the integer guard covers the sums of their synapses.
    counted = []
    for module in layer.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda _module, _inputs, counts: counted.append(counts.abs().sum()))
    layer(spikes)
    assert len(counted) == 4 and fibres.abs().sum() in counted
    model._check_accumulator()
    with torch.no_grad():
        layer.relay.weight.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        model._check_accumulator()
    for relays in (23, -1, 30.0):
        with pytest.raises(ValueError, match="relays"):
            _granules(granule_relays=relays)


def _held_by_bytes(carried, beats, orders, hold, cap):
    """The beat rule one byte at a time: a synapse keeps its stretch, and a beat moves it one place down the chain."""
    batch, steps, clusters, fibres = carried.shape
    heard = torch.zeros_like(carried)
    nothing = (torch.zeros(fibres, dtype=torch.int64), 0, torch.zeros(fibres, dtype=torch.int64))
    for row in range(batch):
        for cluster in range(clusters):
            present, chain = nothing, [None] * int(orders.max())
            for step in range(steps):
                if beats[row, step, cluster]:
                    chain, present = ([present] + chain)[:len(chain)], nothing
                arrived = carried[row, step, cluster].to(torch.int64)
                present = (present[0] + arrived, present[1] + 1, arrived)
                for fibre in range(fibres):
                    stretch = ([present] + chain)[orders[cluster, fibre]]
                    if stretch is None or not stretch[1]:
                        continue
                    total, length, last = stretch[0][fibre], stretch[1], stretch[2][fibre]
                    if hold == "sum":
                        heard[row, step, cluster, fibre] = total.clamp(-cap, cap)
                    elif hold == "last":
                        heard[row, step, cluster, fibre] = last
                    else:
                        # A mean in whole pulses, with halves away from zero.
                        heard[row, step, cluster, fibre] = total.sign() * ((2 * total.abs() + length) // (2 * length))
    return heard


def _beat_case(seed=4, rate=0.3):
    generator = torch.Generator().manual_seed(seed)
    carried = torch.randint(-20, 21, (2, 40, 3, 10), generator=generator).float()
    beats = (torch.rand(2, 40, 3, generator=generator) < rate).float()
    orders = torch.randint(0, 4, (3, 10), generator=generator)
    return carried, beats, orders


@pytest.mark.parametrize("hold", ["mean", "sum", "last"])
def test_beat_fibres_hold_the_stretches_between_the_beats(hold):
    carried, beats, orders = _beat_case()
    # One clock never beats, one beats at the first byte, and the largest sums pass the cap.
    beats[0, :, 0], beats[1, 0, 1] = 0, 1
    heard = _beat_views(carried, beats, orders, hold, 30)
    assert torch.equal(heard, _held_by_bytes(carried, beats, orders, hold, 30))
    assert torch.equal(heard, heard.round()) and heard.abs().max() <= 30
    if hold == "sum":
        assert heard.abs().max() == 30 and _beat_views(carried, beats, orders, hold, 512).abs().max() > 30
    # Without a beat nothing is held, and the present stretch is everything so far.
    assert heard[0, :, 0][:, orders[0] > 0].abs().sum() == 0
    if hold == "sum":
        assert torch.equal(heard[0, :, 0][:, orders[0] == 0], carried[0, :, 0][:, orders[0] == 0].cumsum(0).clamp(-30, 30))
    # What a fibre brings of a held stretch stays the same until the next beat.
    quiet = (beats[:, 1:] == 0)[..., None] & (orders > 0)
    assert torch.equal(heard[:, 1:][quiet], heard[:, :-1][quiet])
    # A fibre hears nothing that comes later.
    later = carried.clone()
    later[:, 25:] += 3
    assert torch.equal(_beat_views(later, beats, orders, hold, 30)[:, :25], heard[:, :25])
    # The running sums are exact up to a length of the byte sequence.
    if hold != "last":
        with pytest.raises(ValueError, match="exact running sums"):
            _beat_views(carried, beats, orders, hold, 1 << 20)


@pytest.mark.parametrize("hold", ["mean", "sum", "last"])
def test_a_clock_that_beats_at_every_byte_makes_delayed_fibres(hold):
    carried, _, orders = _beat_case()
    delayed = torch.stack([F.pad(carried, (0, 0, 0, 0, delay, 0))[:, :40] for delay in range(4)], dim=-1)
    delayed = delayed.gather(-1, orders.expand(2, 40, -1, -1)[..., None]).squeeze(-1)
    assert torch.equal(_beat_views(carried, torch.ones(2, 40, 3), orders, hold, 512), delayed)
    # In a layer the beat fibres then bring their channels as delayed fibres do, beside the layer's other fibres.
    model = _granules(granule_gain=1.0, granule_rest=0.0, granule_delays=2, granule_beats=3, granule_clock="every",
                      granule_hold=hold)
    layer, plain = model.granule_layers["0"], _granules(granule_delays=2).granule_layers["0"]
    assert torch.equal(layer.inputs, plain.inputs) and torch.equal(layer.outputs, plain.outputs)
    # Two delays, the present stretch and three held ones share a cluster's fibres evenly.
    assert all(torch.equal((layer.taps == view).sum(-1), torch.full((6,), 4)) for view in range(6))
    assert layer.beat_fibres.shape == (6, 16) and torch.equal(layer.beat_orders.sort(-1).values,
                                                               torch.arange(4).repeat_interleave(4).expand(6, -1))
    spikes = _stream()
    beats, rates = layer.beats(spikes)
    assert torch.equal(beats, torch.ones(2, 40, 6)) and rates is None
    heard = layer.heard(spikes, beats)
    delay = torch.where(layer.taps < 2, layer.taps, layer.taps - 2)
    for bytes_ago in range(4):
        earlier = F.pad(spikes, (0, 0, bytes_ago, 0))[:, :40][..., layer.inputs]
        assert torch.equal(heard[..., delay == bytes_ago], earlier[..., delay == bytes_ago])
    with pytest.raises(ValueError, match="need the beats"):
        layer.heard(spikes)


def test_the_word_clock_beats_where_a_word_starts():
    text = "An  elephant's na\u00efve 2nd re-run.".encode()
    tokens = torch.tensor([list(text), list(text[1:] + b" ")])
    model = _granules(granule_gain=1.0, granule_rest=0.0, granule_beats=2, granule_hold="mean")
    layer = model.granule_layers["0"]
    spikes = torch.randint(-20, 21, (2, len(text), 96), generator=torch.Generator().manual_seed(6)).float()
    beats, rates = layer.beats(spikes, tokens)
    assert rates is None and beats.shape == (2, len(text), 6) and (beats == beats[..., :1]).all()
    # Letters, digits and the bytes of longer characters make up words. A window that opens inside a word starts one.
    starts = [0, 4, 13, 15, 22, 26, 29]
    assert beats[0, :, 0].nonzero().flatten().tolist() == starts
    assert beats[1, :, 0].nonzero().flatten().tolist() == [0] + [start - 1 for start in starts[1:]]
    with pytest.raises(ValueError, match="bytes themselves"):
        layer.beats(spikes)
    # During "2nd " a fibre of the first order brings its channel's mean over "na\u00efve " and one of the second order
    # the mean over "s ", although those stretches are seven and two bytes long.
    heard = layer.heard(spikes, beats)

    def mean(first, end):
        total, length = spikes[0, first:end].sum(0).to(torch.int64), end - first
        return (total.sign() * ((2 * total.abs() + length) // (2 * length))).float()

    for cluster in range(6):
        place, order = layer.beat_fibres[cluster], layer.beat_orders[cluster]
        channel = layer.inputs[cluster, place]
        for step in range(22, 26):
            brought = heard[0, step, cluster, place]
            assert torch.equal(brought[order == 1], mean(15, 22)[channel[order == 1]])
            assert torch.equal(brought[order == 2], mean(13, 15)[channel[order == 2]])
            assert torch.equal(brought[order == 0], mean(22, step + 1)[channel[order == 0]])
    # The decoder passes its bytes to the clocks.
    assert torch.equal(model(tokens), _decoder()(tokens))
    assert not torch.equal(_grow(model)(tokens), _decoder()(tokens))


def test_the_end_clock_beats_at_the_first_byte_after_a_word():
    text = "An  elephant's na\u00efve 2nd re-run.".encode()
    tokens = torch.tensor([list(text)])
    model = _granules(granule_gain=1.0, granule_rest=0.0, granule_beats=2, granule_clock="end")
    layer = model.granule_layers["0"]
    spikes = torch.randint(-3, 4, (1, len(text), 96), generator=torch.Generator().manual_seed(6)).float()
    beats, rates = layer.beats(spikes, tokens)
    assert rates is None and beats.shape == (1, len(text), 6) and (beats == beats[..., :1]).all()
    assert beats[0, :, 0].nonzero().flatten().tolist() == [2, 12, 14, 21, 25, 28, 32]
    with pytest.raises(ValueError, match="bytes themselves"):
        layer.beats(spikes)
    # From the gap after "2nd" and from inside "re", a fibre of the first order brings the sum over " 2nd" and one of
    # the second order the sum over " na\u00efve": a held stretch is the same word from both places.
    heard = layer.heard(spikes, beats)
    for cluster in range(6):
        place, order = layer.beat_fibres[cluster], layer.beat_orders[cluster]
        channel = layer.inputs[cluster, place]
        for step in range(25, 28):
            brought = heard[0, step, cluster, place]
            assert torch.equal(brought[order == 1], spikes[0, 21:25].sum(0)[channel[order == 1]])
            assert torch.equal(brought[order == 2], spikes[0, 14:21].sum(0)[channel[order == 2]])
            assert torch.equal(brought[order == 0], spikes[0, 25:step + 1].sum(0)[channel[order == 0]])
    assert torch.equal(model(tokens), _decoder()(tokens))
    assert not torch.equal(_grow(model)(tokens), _decoder()(tokens))


def test_clock_cells_beat_at_their_pace_until_they_learn_what_to_listen_to():
    model = _granules(granule_gain=1.0, granule_rest=0.0, granule_beats=3, granule_clock="learned",
                      granule_clock_rate=0.25)
    layer = model.granule_layers["0"]
    cells = layer.beat_fibres.shape[1]
    assert layer.clock.weight.shape == (6, 96) and layer.clock.weight.abs().sum() == 0 and cells == 19
    spikes = torch.randint(-20, 21, (2, 400, 96), generator=torch.Generator().manual_seed(8)).float()
    with torch.no_grad():
        beats, rates = layer.beats(spikes)
    # Without a gradient no rates are kept. Every beat fibre has a clock cell, which fires at most one whole spike
    # per byte.
    assert rates is None and beats.shape == (2, 400, 6, cells) and set(beats.unique().tolist()) == {0.0, 1.0}
    # Deaf clock cells beat at their starting pace, on the grid of quarter spikes of rest: each at a quarter of the
    # bytes, and at every byte a quarter of a cluster's cells, to within one cell.
    pace = torch.sigmoid(torch.tensor(-1.0))
    assert torch.allclose(beats.mean((0, 1)), pace.expand(6, cells), atol=0.02)
    assert (beats.sum(-1) - pace * cells).abs().max() <= 1
    assert not torch.equal(beats[..., 0], beats[..., 1]) and not torch.equal(beats[:, :, 0], beats[:, :, 1])
    assert torch.equal(beats[0], beats[1])
    quick = _granules(granule_beats=3, granule_clock="learned", granule_clock_rate=0.9).granule_layers["0"]
    with torch.no_grad():
        assert torch.allclose(quick.beats(spikes)[0].mean((0, 1, 3)), torch.sigmoid(torch.tensor(2.25)).expand(6),
                              atol=0.01)
    # Clock cells that hear the stream beat where their charge is high: every threshold multiplies the odds by e.
    with torch.no_grad():
        layer.clock.weight[0, :48], layer.clock_rest[0] = 1.5, 0.0
        heard, _ = layer.beats(spikes)
        quanta = 4 + layer.pool(spikes.abs().sum(-1, keepdim=True))
        # A clock cell's threshold is sixteen times a stream neuron's 256 ticks, and a quantum is a quarter of it.
        odds = torch.sigmoid(spikes[..., :48].sum(-1) * 384 / (quanta[..., 0] * 1024))
        likely, unlikely = odds > 0.99, odds < 0.01
        assert likely.sum() > 50 and unlikely.sum() > 50
        assert heard[:, :, 0][likely].mean() > 0.97 and heard[:, :, 0][unlikely].mean() < 0.03
        # At every byte as many of the cluster's cells beat as its rate says, to within one cell. So unsure cells
        # move a part of the fibres on, and sure cells move the whole cluster.
        assert (heard[:, :, 0].mean(-1) - odds).abs().max() <= 1 / cells + 1e-6
        assert torch.equal(heard[:, :, 1:], beats[:, :, 1:])
    # With a gradient the clusters' rates come along, and the beats themselves carry none.
    beats, rates = layer.beats(spikes)
    assert not beats.requires_grad and rates.requires_grad and rates.shape == (2, 400, 6)
    assert torch.allclose(rates[..., 0], odds) and torch.allclose(rates[..., 1:], pace.expand(2, 400, 5))
    # The beats are spikes of the clock cells, which a pulse budget counts.
    counted = []
    layer.tick.register_forward_hook(lambda _module, _inputs, fired: counted.append(emitted_pulses(_module, fired)))
    beats, rates = layer.beats(spikes)
    assert len(counted) == 1 and counted[0] == beats.sum() and counted[0].requires_grad


def test_every_fibre_follows_its_own_beats():
    carried, beats, orders = _beat_case(seed=12)
    own = (torch.rand(2, 40, 3, 10, generator=torch.Generator().manual_seed(13)) < 0.3).float()
    for hold in ("sum", "mean", "last"):
        heard = _beat_views(carried, own, orders, hold, 30)
        # A fibre with its own clock hears what it would hear if the whole cluster kept that clock's time.
        for fibre in range(10):
            alone = _beat_views(carried, own[..., fibre], orders, hold, 30)
            assert torch.equal(heard[..., fibre], alone[..., fibre])
        # Clocks that agree are one clock.
        assert torch.equal(_beat_views(carried, beats[..., None].expand(-1, -1, -1, 10), orders, hold, 30),
                           _beat_views(carried, beats, orders, hold, 30))


@pytest.mark.parametrize("hold", ["sum", "last", "mean"])
def test_clock_cells_learn_from_the_means_over_their_beats(hold):
    carried, beats, orders = _beat_case(seed=9)
    carried, kind = carried.double(), "last" if hold == "last" else "sum"

    def exact(clock, counts=carried):
        return _beat_views(counts, clock.double(), orders, kind, 1 << 14)

    # At rates of zero and one the means over beats are what the beats themselves give: the sums or the last counts
    # of the stretches, and for a mean also their lengths in bytes.
    values, lengths = _expected_beat_views(carried, beats.double().clamp(1e-9, 1 - 1e-6), orders, hold)
    assert torch.allclose(values, exact(beats), atol=1e-3) and (lengths is None) == (hold != "mean")
    if hold == "mean":
        assert torch.allclose(lengths, exact(beats, torch.ones_like(carried)), atol=1e-3)
    # The slope with respect to one beat is what adding or removing that beat changes.
    weights = torch.randn(carried.shape, generator=torch.Generator().manual_seed(10), dtype=torch.float64)
    rates = beats.double().clamp(1e-7, 1 - 1e-5).requires_grad_()
    values, lengths = _expected_beat_views(carried, rates, orders, hold)
    (values * weights).sum().backward(retain_graph=hold == "mean")
    places = ((0, 0, 0), (0, 7, 1), (1, 19, 2), (1, 39, 0), (0, 23, 2))

    def changes(counts):
        for row, step, cluster in places:
            with_beat, without = beats.clone(), beats.clone()
            with_beat[row, step, cluster], without[row, step, cluster] = 1, 0
            yield ((exact(with_beat, counts) - exact(without, counts)) * weights).sum()

    for (row, step, cluster), change in zip(places, changes(carried), strict=True):
        assert torch.allclose(rates.grad[row, step, cluster], change, rtol=1e-3, atol=1e-2)
    if hold == "mean":
        rates.grad = None
        (lengths * weights).sum().backward()
        for (row, step, cluster), change in zip(places, changes(torch.ones_like(carried)), strict=True):
            assert torch.allclose(rates.grad[row, step, cluster], change, rtol=1e-3, atol=1e-2)
    # They are the means over clocks that beat with these chances.
    generator = torch.Generator().manual_seed(11)
    chances = torch.rand(1, 40, 3, generator=generator, dtype=torch.float64) * 0.6
    drawn = (torch.rand(4000, 40, 3, generator=generator, dtype=torch.float64) < chances).double()
    sampled = exact(drawn, carried[:1].expand(4000, -1, -1, -1)).mean(0)
    assert (sampled - _expected_beat_views(carried[:1], chances, orders, hold)[0][0]).abs().max() < 1.5
    # The whole beats decide the values, and the rates only add a slope. A mean's slope is that of its sum and its
    # length at the stretch as it is: more bytes in a stretch pull its mean toward zero.
    rates = torch.full((2, 40, 3), 0.3, dtype=torch.float64).requires_grad_()
    views = _beat_views(carried, beats.double(), orders, hold, 1 << 14, rates)
    assert torch.equal(views, _beat_views(carried, beats.double(), orders, hold, 1 << 14))
    (views * weights).sum().backward()
    assert rates.grad.abs().sum() > 0 and torch.isfinite(rates.grad).all()


def test_a_learned_clock_gets_its_slope_through_the_cells_and_only_in_training():
    model = _grow(_granules(granule_gain=1.0, granule_rest=0.0, granule_budget=6, granule_beats=3,
                            granule_clock="learned"))
    layer = model.granule_layers["0"]
    spikes = _stream().requires_grad_()
    layer(spikes).square().sum().backward()
    assert layer.clock.weight.grad.abs().sum() > 0 and layer.clock_rest.grad.abs().sum() > 0
    assert torch.isfinite(layer.clock.weight.grad).all() and spikes.grad.abs().sum() > 0
    # The whole spikes decide what the cells hear: the mean over beats adds nothing to the values.
    with torch.no_grad():
        silent = layer(spikes)
    assert torch.equal(layer(spikes), silent)
    # The pool, the cells, the clocks and the stream's firing population report their pulses.
    counted = []
    with torch.no_grad():
        beats, cells = layer.beats(spikes)[0].sum(), layer.cells(spikes).sum()
    for module in layer.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda _module, _inputs, counts: counted.append(counts.abs().sum()))
    with torch.no_grad():
        layer(spikes)
    assert len(counted) == 4 and beats > 0 and beats in counted and cells in counted and beats != cells
    # The integer guard sees the clock cells' synapses.
    model._check_accumulator()
    with torch.no_grad():
        layer.clock.weight.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        model._check_accumulator()


def test_beat_fibres_leave_other_layers_and_old_weights_as_they_were():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    plain = _decoder()
    for clock in ("word", "end", "every", "learned"):
        model = _granules(granule_beats=4, granule_clock=clock, granule_delays=2)
        assert torch.equal(model(tokens), plain(tokens))
        assert not torch.equal(_grow(model)(tokens), plain(tokens))
    # A layer without beat fibres has the wiring and the weights it had before there were any.
    old, new = _granules(granule_delays=4), _granules(granule_delays=4, granule_beats=0, granule_clock="learned")
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(value, new.state_dict()[name]) for name, value in old.state_dict().items())
    assert not any("clock" in name or "beat" in name for name in old.state_dict())
    learned = _granules(granule_delays=4, granule_beats=2, granule_clock="learned").state_dict()
    assert set(learned) - set(old.state_dict()) == {f"granule_layers.{index}.{name}" for index in (0, 2)
                                                    for name in ("clock.weight", "clock_rest")}


def test_the_axons_of_a_cluster_can_reach_more_or_fewer_channels_than_it_hears():
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    same, named = _grow(_granules(granule_budget=3)), _grow(_granules(granule_budget=3, granule_reach=24))
    # A reach equal to the fan is the layer as it was.
    assert same.state_dict().keys() == named.state_dict().keys()
    assert torch.equal(same(tokens), named(tokens))
    for reach in (8, 96):
        model = _grow(_granules(granule_budget=3, granule_reach=reach))
        for layer in model.granule_layers.values():
            assert layer.inputs.shape == (6, 24) and layer.receive.shape == (6, 20, 24)
            assert layer.outputs.shape == (6, reach) and layer.send.shape == (6, reach, 20)
            assert all(len(set(row.tolist())) == reach for row in layer.outputs)
        counts = model(tokens)
        assert counts.shape == (2, 80, 256) and torch.isfinite(counts).all()
        counts.square().mean().backward()
        for layer in model.granule_layers.values():
            assert torch.isfinite(layer.send.grad).all() and layer.send.grad.abs().sum() > 0
            assert torch.isfinite(layer.receive.grad).all() and layer.receive.grad.abs().sum() > 0
        # The cells write to the channels their axons reach and to no others.
        layer, stream = model.granule_layers["0"], _stream()
        reached = torch.zeros(96, dtype=torch.bool)
        reached[layer.outputs.flatten()] = True
        with torch.no_grad():
            layer.send.mul_(40)
        changed = (layer(stream) != stream).flatten(0, 1).any(0)
        assert changed.any() and not changed[~reached].any()
    for reach in (0, 97, 2.0):
        with pytest.raises(ValueError, match="reach a whole number"):
            _granules(granule_reach=reach)


def test_granule_settings_are_checked():
    with pytest.raises(ValueError, match="byte-rate blocks"):
        _granules(granule_after=(4,))
    with pytest.raises(ValueError, match="byte-rate blocks"):
        _granules(granule_after=(1, 1))
    with pytest.raises(ValueError, match="positive sizes"):
        _granules(granule_fan=97)
    with pytest.raises(ValueError, match="positive sizes"):
        _granules(granule_cells=0)
    with pytest.raises(ValueError, match="positive sizes"):
        _granules(granule_rest=17.0)
    with pytest.raises(ValueError, match="positive sizes"):
        _granules(granule_price=1.5)
    for settings in (dict(granule_beats=9), dict(granule_beats=2.0), dict(granule_beats=2, granule_clock="tick"),
                     dict(granule_beats=2, granule_hold="most"), dict(granule_beats=2, granule_clock_rate=1.0),
                     dict(granule_beats=2, granule_clock_rate=0.0)):
        with pytest.raises(ValueError, match="beat fibres hold"):
            _granules(**settings)
    with pytest.raises(ValueError, match="each held stretch"):
        _granules(granule_fan=6, granule_delays=2, granule_traces=1, granule_beats=3)
    model = _granules()
    with pytest.raises(ValueError, match="carried stream"):
        model.forward_chunk(torch.zeros(1, 8, dtype=torch.long))
