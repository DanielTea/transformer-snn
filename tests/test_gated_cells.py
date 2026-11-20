import pytest
import torch

from snnformer.event_lm import (_GATE_CAP, EventLM, EventLMConfig, _ExcitatoryFire, _GatedFire, _IntegrateFire,
                                emitted_pulses)

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=64, granule_slope=True, granule_gate=4)
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


def test_a_gated_cell_has_a_gate_dendrite_beside_its_value_dendrite():
    plain, gated = _stack(granule_gate=0, granule_cells=96), _stack()
    extra = set(gated.state_dict()) - set(plain.state_dict())
    assert extra == {f"granule_layers.{level}.gate" for level in _LEVELS}
    layer = gated.granule_layers["0"]
    assert layer.gate.shape == layer.receive.shape == (1, 64, 64) and layer.rest.shape == (1, 64, 1)
    # Three rows of synapses a cell, where a plain cell has two: 64 gated cells cost what 96 plain cells cost.
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    assert count(gated) - count(plain) == len(_LEVELS) * (64 - 96)


def test_the_body_fires_for_the_charge_that_the_gate_lets_in():
    fire = _GatedFire(128, 8, 4)
    quanta = torch.full((1,), 4.0)
    # At rest the pool gives four quanta, and a threshold is 256 ticks. Four gate spikes pass the charge once.
    value = torch.tensor([512.0, -512.0, 300.0, 127.0, 1000.0])
    assert fire(value, quanta, opened=torch.full((5,), 4.0)).tolist() == [2, -2, 1, 0, 4]
    # A closed gate passes nothing, two spikes pass half, and eight pass the charge twice.
    assert fire(value, quanta, opened=torch.zeros(5)).tolist() == [0, 0, 0, 0, 0]
    assert fire(value, quanta, opened=torch.full((5,), 2.0)).tolist() == [1, -1, 1, 0, 2]
    assert fire(value, quanta, opened=torch.full((5,), 8.0)).tolist() == [4, -4, 2, 1, 8]
    # The pool's inhibition raises the threshold: with eight quanta the same charge counts half.
    assert fire(value, torch.full((1,), 8.0), opened=torch.full((5,), 4.0)).tolist() == [1, -1, 1, 0, 2]
    # The collected charge is bounded like a plain cell's, and the count ends at the cell's largest.
    large = torch.tensor([10.0 ** 7, -(10.0 ** 7)])
    assert fire(large, quanta, opened=torch.full((2,), 64.0)).tolist() == [128, -128]
    # Whole numbers in, whole numbers out.
    whole = fire(value.to(torch.int32), quanta.to(torch.int32), opened=torch.full((5,), 4, dtype=torch.int32))
    assert whole.dtype == torch.int32 and whole.tolist() == [2, -2, 1, 0, 4]


def test_the_body_learns_through_both_dendrites():
    fire = _GatedFire(128, 8, 4)
    quanta = torch.full((1,), 4.0)
    value = torch.tensor([512.0, -300.0, 40.0], requires_grad=True)
    opened = torch.tensor([4.0, 2.0, 0.0], requires_grad=True)
    fire(value, quanta, opened=opened).sum().backward()
    # The count's slope is that of the product: the gate's spikes over a threshold, and the charge over it.
    torch.testing.assert_close(value.grad, torch.tensor([4.0, 2.0, 0.0]) / 1024)
    torch.testing.assert_close(opened.grad, torch.tensor([512.0, -300.0, 40.0]) / 1024)


def test_gated_cells_fire_whole_spikes_on_both_rails_and_only_where_the_gate_opens():
    model, tokens = _stack(), _bytes()
    layer = model.granule_layers["0"]
    stream = _stream(model, tokens)
    cells = layer.cells(stream)
    assert cells.shape == (2, 80, 1, 64) and torch.equal(cells, cells.round())
    assert (cells > 0).any() and (cells < 0).any()
    # Rebuild the gate's spikes by hand.
    heard = layer.heard(stream)
    quanta = (4 + layer.pool(stream.abs().sum(-1, keepdim=True)))[..., None]
    rest = layer.rest_quarters().squeeze(-1) * quanta * 16
    opened = layer.open(torch.einsum("btgk,gnk->btgn", heard, (layer.gate * 256).round()) - rest, quanta)
    assert torch.equal(opened, opened.round()) and opened.min() == 0 and 0 < (opened > 0).float().mean() < 1
    assert not cells[opened == 0].any()
    # Four gate spikes are a threshold of the gate, and at most sixteen thresholds count.
    assert opened.max() <= _GATE_CAP * 4
    # With every gate shut by a large rest the layer is silent and passes the stream on.
    with torch.no_grad():
        layer.rest.fill_(16.0)
    assert not layer.cells(stream).any() and torch.equal(layer(stream), stream)


def test_a_stack_of_gated_cells_fires_whole_spikes_and_sees_no_later_byte():
    model, tokens = _stack(), _bytes()
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])


def test_every_synapse_of_a_gated_stack_learns_and_so_does_a_shut_gate():
    model = _stack()
    model(_bytes()).float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    # A gate that never opens on these bytes still has a slope, as a silent plain cell has with its smooth threshold.
    model = _stack()
    layer = model.granule_layers["0"]
    with torch.no_grad():
        layer.rest[0, :8] = 5.0
    tokens = _bytes()
    stream = _stream(model, tokens)
    assert not layer.cells(stream)[..., :8].any()
    model(tokens).float().square().mean().backward()
    assert layer.gate.grad[0, :8].abs().sum() > 0 and layer.rest.grad[0, :8].abs().sum() > 0


def test_gated_cells_stay_within_the_range_of_exact_sums():
    model = _stack()
    model._check_accumulator()
    layer = model.granule_layers["1"]
    value, _ = layer.accumulator_bounds()
    with torch.no_grad():
        # The gate dendrite sums its synapses and the resting charge, the value dendrite its synapses alone.
        layer.gate.fill_(1.9)
    gate, _ = layer.accumulator_bounds()
    assert gate > value and gate == 64 * 486 + 16 * 4 * (4 + layer.pool.cap) * 16 // 512 + (
        16 * 4 * (4 + layer.pool.cap) * 16 % 512 > 0)
    with torch.no_grad():
        layer.gate.fill_(0.0)
        layer.receive.fill_(1.9)
    assert layer.accumulator_bounds()[0] == 64 * 486
    # The charge a gate lets in at its largest count is a whole number that FP32 holds exactly.
    assert layer.fire.threshold * layer.fire.cap * _GATE_CAP * layer.gated <= 1 << 24
    with pytest.raises(ValueError, match="gated"):
        _stack(granule_gate=8, granule_bits=10)


def test_the_pulses_of_gated_cells_are_counted_at_the_gate_and_at_the_body():
    model, tokens = _stack(), _bytes()
    fired = []
    for module in model.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda population, _, counts: fired.append(emitted_pulses(population, counts)))
    model(tokens)
    # The input, the output, and in every level the read's pool, probes, tags, payloads and write, and the cells'
    # pool, their gates, their bodies and their write.
    assert len(fired) == 2 + len(_LEVELS) * 9 and all(count >= 0 for count in fired)


@pytest.mark.parametrize("setting", [dict(granule_gate=-1), dict(granule_gate=2.0),
                                     dict(granule_budget=8, granule_slope=False), dict(granule_gate=64)])
def test_a_layer_refuses_gates_it_cannot_have(setting):
    with pytest.raises(ValueError, match="gated"):
        _stack(**setting)


def _quanta(count):
    return torch.full((1,), float(count))


def test_gate_and_body_count_in_the_quanta_of_a_fine_pool():
    body = _GatedFire(128, 8, 4, 64)
    value = torch.tensor([512.0, -512.0, 300.0, 127.0, 1000.0])
    # Sixty-four quanta are a whole threshold of 256 ticks, where four gate spikes pass the charge once.
    assert body(value, _quanta(64), opened=torch.full((5,), 4.0)).tolist() == [2, -2, 1, 0, 4]
    assert body(value, _quanta(64), opened=torch.full((5,), 2.0)).tolist() == [1, -1, 1, 0, 2]
    # Four quanta are a sixteenth of a threshold and the same charge counts sixteen times. Twice the quanta halve it.
    assert body(value, _quanta(4), opened=torch.full((5,), 4.0)).tolist() == [32, -32, 19, 8, 63]
    assert body(value, _quanta(128), opened=torch.full((5,), 4.0)).tolist() == [1, -1, 1, 0, 2]
    # A body fires its largest count at most, and fewer spikes once its threshold is above a whole one.
    large, wide = torch.tensor([10.0 ** 7, -(10.0 ** 7)]), torch.full((2,), 64.0)
    assert body(large, _quanta(4), opened=wide).tolist() == [128, -128]
    assert body(large, _quanta(128), opened=wide).tolist() == [64, -64]
    whole = body(value.to(torch.int32), _quanta(4).to(torch.int32), opened=torch.full((5,), 4, dtype=torch.int32))
    assert whole.dtype == torch.int32 and whole.tolist() == [32, -32, 19, 8, 63]
    # The gate fires four spikes for a threshold of charge whatever the threshold is. It counts sixteen whole
    # thresholds of charge at most, and no more spikes than those are at a whole threshold.
    gate = _ExcitatoryFire(_GATE_CAP, 8, False, quanta_per_threshold=256, capped=_GATE_CAP * 4)
    charge = torch.tensor([256.0, 300.0, 10.0 ** 6, 16.0, -500.0, 33.0])
    assert gate(charge, _quanta(64)).tolist() == [4, 5, 64, 0, 0, 1]
    assert gate(charge, _quanta(4)).tolist() == [64, 64, 64, 4, 0, 8]
    assert gate(charge, _quanta(512)).tolist() == [1, 1, 8, 0, 0, 0]
    # Under a block's pool a threshold is never below a whole one, and the gate is what it was.
    capped = _ExcitatoryFire(_GATE_CAP, 8, False, quanta_per_threshold=16, capped=_GATE_CAP * 4)
    free = _ExcitatoryFire(_GATE_CAP, 8, False, quanta_per_threshold=16)
    charges = torch.randint(-5000, 5000, (2000,), generator=torch.Generator().manual_seed(3)).float()
    assert all(torch.equal(capped(charges, _quanta(count)), free(charges, _quanta(count))) for count in range(4, 133))


def test_a_fine_pool_holds_the_firing_of_gated_cells_when_the_stream_grows():
    tokens, growth = _bytes(), {}
    for name, settings in (("block", {}), ("fine", _FINE)):
        model = _stack(**settings)
        layer, stream = model.granule_layers["0"], _stream(model, tokens)
        once, thrice = (layer.cells(scale * stream).abs().mean() for scale in (1, 3))
        growth[name] = (thrice / once).item()
    # A body counts the product of two charges. With three times the pulses it fires nearly three times as much under
    # a block's pool, and as before under a fine pool.
    assert growth["block"] > 2.5 and 0.95 < growth["fine"] < 1.05


def test_a_gated_stack_with_fine_pools_fires_whole_spikes_learns_and_stays_exact():
    model, tokens = _stack(**_FINE), _bytes()
    layer = model.granule_layers["0"]
    assert (layer.open.quanta_per_threshold, layer.open.capped, layer.fire.quanta_per_threshold) == (256, 64, 64)
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
    cells = layer.cells(_stream(model, tokens))
    assert torch.equal(cells, cells.round()) and (cells > 0).any() and (cells < 0).any()
    first.float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    model._check_accumulator()
    # The gate dendrite sums its synapses and sixteen spikes of rest, each a quarter of a quantum of one tick, at the
    # most quanta, per unit of a stream neuron's 512 spikes.
    with torch.no_grad():
        layer.gate.fill_(1.9)
    assert layer.accumulator_bounds()[0] == 64 * 486 + -(-16 * 4 * (4 + 2048) // 512)
    # The input, the output and its pool, and in every level the nine populations of a gated stack.
    fired = []
    for module in model.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda population, _, counts: fired.append(emitted_pulses(population, counts)))
    model(tokens)
    assert len(fired) == 3 + len(_LEVELS) * 9 and all(count >= 0 for count in fired)


def test_a_gate_spike_under_a_fine_pool_is_a_whole_number_of_ticks():
    with pytest.raises(ValueError, match="gated"):
        _stack(granule_gate=8, **_FINE)
