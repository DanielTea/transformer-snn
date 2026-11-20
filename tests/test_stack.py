import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _DelayBlock, _IntegrateFire, _Level, emitted_pulses

_LEVELS = (0, 1, 2)
_STACK = dict(dim=64, hidden=4, depth=3, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True)


def _stack(seed=0, grown=True, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    if grown:
        # Reads and cell layers start silent. Give them output synapses, and the heads an address.
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            for read in (model.state_reads or {}).values():
                read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
                read.age.copy_(torch.randn(read.age.shape, generator=generator) * 0.2)
            for layer in (model.granule_layers or {}).values():
                layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=80):
    return torch.randint(0, 256, (2, length), generator=torch.Generator().manual_seed(seed))


def test_a_stack_has_levels_and_no_blocks():
    model = _stack()
    assert all(isinstance(level, _Level) for level in model.blocks)
    assert not any(isinstance(module, _DelayBlock) for module in model.modules())
    # Every parameter belongs to the input, the output, a state read or a cell layer.
    parts = {name.split(".")[0] for name, _ in model.named_parameters()}
    assert parts == {"input", "output", "state_reads", "granule_layers"}
    # A level passes the stream on as it is.
    stream = torch.randint(-9, 10, (2, 5, 64)).float()
    assert model.blocks[0](stream, delay_limit=None, pulse_counts=None) is stream


def test_a_stack_is_its_reads_and_cells_in_order():
    model, tokens = _stack(), _bytes()
    spikes = model.input_emit(model.input(tokens) * model.cfg.resolution)
    for level in _LEVELS:
        spikes = model.state_reads[str(level)](spikes, None, tokens)
        spikes = model.granule_layers[str(level)](spikes, tokens)
    current = model.output(spikes)
    assert torch.equal(model(tokens), model.output_emit(current, reference=model._reference_counts(current)))


def test_a_silent_stack_reads_the_bytes_alone():
    tokens = _bytes()
    silent = _stack(grown=False)(tokens)
    again = tokens.clone()
    again[:, :-1] = _bytes(seed=5)[:, :-1]
    # Nothing has grown yet, so the last byte's output is the same whatever came before it.
    assert torch.equal(silent[:, -1], _stack(grown=False)(again)[:, -1])
    assert not torch.equal(_stack()(tokens)[:, -1], _stack()(again)[:, -1])


def test_a_stack_fires_whole_spikes_and_sees_no_later_byte():
    model, tokens = _stack(), _bytes()
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50])
    assert not torch.equal(first[:, 50:], second[:, 50:])


def test_every_synapse_of_a_stack_learns():
    model = _stack()
    model(_bytes()).float().square().mean().backward()
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())


def test_a_stack_stays_within_the_range_of_exact_sums():
    model = _stack()
    model._check_accumulator()
    with torch.no_grad():
        # 96 cells at 486 ticks each could bring a channel more charge than whole numbers hold exactly.
        model.granule_layers["1"].send.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        model._check_accumulator()


def test_a_capped_cell_fires_no_more_than_its_cap():
    tokens = _bytes()
    free, capped = _stack(granule_gain=4.0, granule_rest=0.0), _stack(granule_gain=4.0, granule_rest=0.0, granule_cap=8)
    stream = free.input_emit(free.input(tokens) * 16)
    most, few = free.granule_layers["0"].cells(stream), capped.granule_layers["0"].cells(stream)
    # The pool's inhibition raises the thresholds, so a cell's count ends below the cap where the stream is busy.
    assert most.max() > 8 and few.max() <= 8
    assert torch.equal(few, few.round()) and torch.equal(few[most <= 2], most[most <= 2])
    # A channel may then take synapses from its cells that are larger by the same factor.
    with torch.no_grad():
        for model in (free, capped):
            model.granule_layers["0"].send.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        free._check_accumulator()
    capped._check_accumulator()
    loose, tight = (model.granule_layers["0"].accumulator_bounds()[1] for model in (free, capped))
    assert tight - 256 == torch.ceil((loose - 256) * 8 / 512)
    with pytest.raises(ValueError, match="largest count"):
        _stack(granule_cap=513)


def test_a_capped_read_fires_no_more_than_its_cap():
    tokens = _bytes()
    free, capped = _stack(), _stack(read_cap=2)
    stream = free.input_emit(free.input(tokens) * 16)
    for most, few in zip(free.state_reads["0"].packets(stream), capped.state_reads["0"].packets(stream), strict=True):
        # The pool's inhibition raises the thresholds, so the largest count ends below the cap on a busy stream.
        assert most.abs().max() > 2 and 0 < few.abs().max() <= 2
        assert torch.equal(few, few.round()) and torch.equal(few[most.abs() <= 1], most[most.abs() <= 1])
    # The read's output synapses may then be larger by about the same factor: a read neuron brings up to four
    # read pulses for each payload pulse, and half a pulse of rounding.
    loose, tight = (model.state_reads["0"].accumulator_bounds()[3] for model in (free, capped))
    most = free.state_reads["0"].project.weight_ticks().abs().sum(-1).amax()
    assert torch.isclose(loose - 256, most * (4 * 512 + 0.5) / 512) and torch.isclose(tight - 256, most * 8.5 / 512)
    with pytest.raises(ValueError, match="largest count"):
        _stack(read_cap=0)


def test_a_fan_of_several_streams_hears_the_present_and_the_last_bytes():
    model = _stack(granule_fan=128, granule_delays=4)
    layer = model.granule_layers["1"]
    # Every channel has two fibres: one for the present and one that is one, two or three bytes late.
    assert layer.inputs.shape == layer.taps.shape == (1, 128) and layer.outputs.shape == (1, 64)
    assert torch.equal(layer.inputs[0], torch.arange(64).repeat(2))
    assert torch.equal(layer.taps[0, :64], torch.zeros(64, dtype=torch.long))
    assert sorted(torch.bincount(layer.taps[0, 64:]).tolist()) == [0, 21, 21, 22]
    stream = torch.randint(-9, 10, (2, 12, 64), generator=torch.Generator().manual_seed(4)).float()
    heard = layer.heard(stream)[:, :, 0]
    assert torch.equal(heard[..., :64], stream)
    for delay in (1, 2, 3):
        late = layer.taps[0, 64:] == delay
        assert torch.equal(heard[:, delay:, 64:][..., late], stream[:, :-delay][..., late])
        assert heard[:, :delay, 64:][..., late].abs().sum() == 0
    # With as many fibres as delays a cell hears every channel at every delay.
    full = _stack(granule_fan=256, granule_delays=4).granule_layers["0"]
    assert sorted(zip(full.inputs[0].tolist(), full.taps[0].tolist(), strict=True)) == [
        (channel, delay) for channel in range(64) for delay in range(4)]
    assert full.receive.shape == (1, 96, 256) and full.send.shape == (1, 64, 96)
    # The decoder stays causal and whole.
    tokens, changed = _bytes(), _bytes()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round()) and torch.equal(first[:, :50], second[:, :50])
    for setting in (dict(granule_fan=100), dict(granule_fan=192, granule_delays=2), dict(granule_fan=128),
                    dict(granule_fan=128, granule_delays=4, granule_traces=1)):
        with pytest.raises(ValueError):
            _stack(**setting)


def test_the_pulses_of_a_stack_are_counted():
    model, tokens = _stack(), _bytes()
    fired, reads = [], []
    for module in model.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda population, _, counts: fired.append(emitted_pulses(population, counts)))
    model(tokens, memory_pulses=reads)
    # The input, the output, and in every level the read's pool, probes, tags, payloads and write, and the cells'
    # pool, the cells and their write.
    assert len(fired) == 2 + len(_LEVELS) * 8
    # Every read reports its cell spikes and its packet pulses beside them.
    assert len(reads) == 2 * len(_LEVELS) and all(count > 0 for count in reads)


def test_a_stack_may_have_reads_or_cells_alone():
    tokens = _bytes()
    for part in ("read_after", "granule_after"):
        model = _stack(**{part: None})
        assert torch.equal(model(tokens), model(tokens).round())
        assert {name.split(".")[0] for name, _ in model.named_parameters()} == {
            "input", "output", "granule_layers" if part == "read_after" else "state_reads"}


@pytest.mark.parametrize("setting", [dict(carry=True), dict(clock_start=1), dict(route="steered_correction_code"),
                                     dict(read_after=None, granule_after=None), dict(activity_inhibition=True),
                                     dict(local_trace=True, compact_delays=True)])
def test_a_stack_takes_no_option_of_a_block(setting):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **setting)))


def test_the_stepper_refuses_reads_and_cells():
    with pytest.raises(ValueError, match="whole windows"):
        _stack().eval().to_integer().step(torch.tensor([65, 66]))
    # A decoder with blocks and a state read is refused as well, where the stepper used to skip the read.
    blocks = EventLM(EventLMConfig(**dict(_STACK, stack=False, hidden=8, depth=2, read_after=(0,), granule_after=None)))
    with pytest.raises(ValueError, match="whole windows"):
        blocks.eval().to_integer().step(torch.tensor([65]))
