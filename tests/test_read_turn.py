import math

import pytest
import torch

from snnformer.event_lm import (_PHASE_SPIKES, _TURN_SHORTEST, _TURN_SPAN, EventLM, EventLMConfig, _IntegrateFire,
                                _pacemaker_counts, emitted_pulses)

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, read_turn=True, granule_after=_LEVELS, granule_clusters=1,
              granule_fan=64, granule_cells=96, granule_slope=True)


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


def test_the_gating_pacemakers_turn_by_a_radian_a_byte_at_the_fastest():
    read = _stack().state_reads["0"]
    # Four pairs of channels in each of four heads: four pacemaker pairs, which the heads share.
    assert read.turners.shape == (48, 8)
    periods = _TURN_SHORTEST * _TURN_SPAN ** (torch.arange(4, dtype=torch.float64) / 4)
    assert read.turn_longest == pytest.approx(periods[-1].item())
    steps = torch.arange(48, dtype=torch.float64)[:, None]
    exact = torch.cat(((2 * math.pi * steps / periods).cos(), (2 * math.pi * steps / periods).sin()), -1)
    assert torch.equal(read.turners, (exact * _PHASE_SPIKES).round().float())
    # The fastest pair's phase is the number of the byte in radians.
    assert torch.equal(read.turners[:, 0], (torch.arange(48).double().cos() * _PHASE_SPIKES).round().float())
    # A window longer than the read window gets the same rhythms, carried on.
    assert torch.equal(read.turn_counts(100)[:48], read.turners[:48]) and read.turn_counts(100).shape == (100, 8)
    assert torch.equal(read.turn_counts(100), _pacemaker_counts(100, 4, read.turn_longest, _TURN_SHORTEST))


def test_a_turned_read_has_the_synapses_of_a_plain_one():
    plain, turned = _stack(read_turn=False), _stack()
    assert set(plain.state_dict()) == set(turned.state_dict())
    assert all(torch.equal(value, turned.state_dict()[name]) for name, value in plain.state_dict().items())
    assert not hasattr(plain.state_reads["0"], "turners")
    tokens = _bytes()
    assert not torch.equal(plain(tokens), turned(tokens))


def test_the_pacemakers_let_the_charges_of_a_pair_in_turned_by_their_phase():
    read = _stack().state_reads["0"]
    drive = torch.zeros(1, 48, 32)
    # The first pair of the first head holds charge on its first dendrite, the second pair on its second.
    drive[..., 0] = 300.0
    drive[..., 3] = -200.0
    let_in = read.let_in(drive)
    cosine, sine = read.turners[:, :4], read.turners[:, 4:]
    assert torch.equal(let_in[0, :, 0], 300 * cosine[:, 0]) and torch.equal(let_in[0, :, 1], 300 * sine[:, 0])
    assert torch.equal(let_in[0, :, 2], 200 * sine[:, 1]) and torch.equal(let_in[0, :, 3], -200 * cosine[:, 1])
    assert not let_in[0, :, 4:].any()
    # A dendrite holds no more charge than a neuron counts, and the sum stays a whole number that FP32 holds.
    limit = read.fire.threshold * read.fire.cap
    full = read.let_in(torch.full((1, 48, 32), 1e9))
    assert full.abs().max() <= 2 * _PHASE_SPIKES * limit <= 1 << 24
    assert torch.equal(full[0, :, 0], limit * (cosine[:, 0] - sine[:, 0]))
    # At the pacemaker's largest count a threshold of a dendrite's charge is one pulse, as without pacemakers.
    quanta = torch.full((1, 1, 1), 4.0)
    assert read.turned(read.let_in(drive), quanta)[0, 0, :4].tolist() == [1, 0, 0, -1]
    assert read.fire(drive, quanta)[0, 0, :4].tolist() == [1, 0, 0, -1]


def test_the_match_of_turned_pulses_tells_the_bytes_between_them():
    read = _stack().state_reads["0"]
    drive = torch.zeros(1, 48, 32)
    drive[..., 0], drive[..., 1] = 3000.0, -1700.0
    quanta = torch.full((1, 1, 1), 4.0)
    pulses = read.turned(read.let_in(drive), quanta)[0, :, :2].double()
    assert torch.equal(pulses, pulses.round()) and (pulses > 0).any() and (pulses < 0).any()
    match = pulses @ pulses.T
    # The same charges at every byte: the match of two bytes follows the cosine of the bytes between them, up
    # to the rounding of the pulses, wherever the two bytes are.
    size = (3000.0 ** 2 + 1700.0 ** 2) / 256 ** 2
    for distance in range(12):
        diagonal = match.diagonal(-distance)
        assert (diagonal - size * math.cos(distance)).abs().max() < 0.12 * size


def test_a_turned_stack_fires_whole_spikes_and_sees_no_later_byte():
    model, tokens = _stack(), _bytes()
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
    probe, tag, payload = model.state_reads["0"].packets(_stream(model, tokens))
    for name, packet in (("probe", probe), ("tag", tag), ("payload", payload)):
        assert torch.equal(packet, packet.round()) and (packet > 0).any() and (packet < 0).any(), name


@pytest.mark.parametrize("distance", [1, 2, 5])
def test_a_probe_turned_back_meets_the_cell_that_many_bytes_back(distance):
    model = _stack(grown=False)
    read = model.state_reads["0"]
    # Every byte the same: all cells hold the same thing and only their ages differ.
    tokens = torch.full((2, 48), 65)
    stream = _stream(model, tokens)
    pairs = read.turners.shape[1] // 2
    turn = 2 * math.pi * distance / (_TURN_SHORTEST * _TURN_SPAN ** (torch.arange(pairs) / pairs))
    with torch.no_grad():
        read.gain.fill_(math.log(8.0) / read.rate)
        # The first dendrite of the two fastest pairs of a tag hears the stream, and nothing else does.
        heard = 0.05 * stream[0, 0].sign()
        tag = torch.zeros(read.heads, pairs, 2, heard.numel())
        tag[:, :2, 0] = heard
        # The dendrites of a probe pair hear what those of the tag pair hear, turned back by the phase that the
        # pair's pacemakers pass in that many bytes.
        cosine, sine = turn.cos()[:, None], turn.sin()[:, None]
        probe = torch.stack((tag[:, :, 0] * cosine + tag[:, :, 1] * sine,
                             tag[:, :, 1] * cosine - tag[:, :, 0] * sine), 2)
        read.tag.weight.copy_(tag.reshape(read.tag.weight.shape))
        read.probe.weight.copy_(probe.reshape(read.probe.weight.shape))
    shares = read.shares(stream, tokens)[0]
    # Every head of every byte puts nearly all of its read on the cell that many bytes back.
    cells = torch.arange(distance, tokens.shape[1])
    assert torch.equal(shares[:, :, distance:].argmax(-1), (cells - distance).expand(2, read.heads, -1))
    assert shares[:, :, distance:].amax(-1).min() > 0.9
    # Without the pacemakers the same synapses tell no cell from another.
    plain = _stack(grown=False, read_turn=False).state_reads["0"]
    plain.load_state_dict(read.state_dict())
    level = plain.shares(stream, tokens)[0]
    assert level[:, :, -1].amax(-1).max() < 0.1


def test_every_synapse_of_a_turned_read_learns():
    model = _stack()
    model(_bytes()).float().square().mean().backward()
    for read in model.state_reads.values():
        for synapse in (read.probe, read.tag, read.payload, read.project):
            assert synapse.weight.grad is not None and synapse.weight.grad.abs().sum() > 0


def test_the_gating_pacemakers_pulses_are_counted_once():
    plain, turned, tokens = _stack(read_turn=False), _stack(), _bytes()
    without, reads = [], []
    plain(tokens, memory_pulses=without)
    turned(tokens, memory_pulses=reads)
    assert len(reads) == len(without) + 1 == 2 * len(_LEVELS) + 1
    assert reads[-1] == 2 * turned.state_reads["0"].turn_counts(80).abs().sum()
    # The turned probes and tags are firing populations like the plain ones.
    fired = []
    for module in turned.modules():
        if isinstance(module, _IntegrateFire):
            module.register_forward_hook(lambda population, _, counts: fired.append(emitted_pulses(population, counts)))
    turned(tokens)
    assert len(fired) == 2 + len(_LEVELS) * 8


def test_a_turned_stack_stays_within_the_range_of_exact_sums():
    model = _stack()
    model._check_accumulator()
    read = model.state_reads["0"]
    assert 2 * _PHASE_SPIKES * read.fire.threshold * read.fire.cap <= 1 << 24
    with pytest.raises(ValueError, match="pacemakers let into"):
        _stack(resolution=32, read_resolution=4)


@pytest.mark.parametrize("setting", [dict(read_mode="soft", read_budget=None), dict(read_signed=False),
                                     dict(read_width=36), dict(read_budget=None)])
def test_a_read_refuses_a_turn_it_cannot_make(setting):
    with pytest.raises(ValueError, match="turn"):
        _stack(**setting)
