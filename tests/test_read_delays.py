import math

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, read_delays=4, granule_after=_LEVELS, granule_clusters=1,
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
    return model.input_emit(model.input(tokens) * 16)


def test_a_read_has_late_fibres_only_when_asked():
    plain, late, both = _stack(read_delays=1), _stack(), _stack(read_delay_probes=True)
    # The late fibres are synapses of the tags, and of the probes where those hear them too. Nothing else is new.
    assert set(plain.state_dict()) == set(late.state_dict()) == set(both.state_dict())
    for model, probe, tag in ((plain, 64, 64), (late, 64, 128), (both, 128, 128)):
        read = model.state_reads["0"]
        assert read.probe.weight.shape == (32, probe) and read.tag.weight.shape == (32, tag)
        assert read.payload.weight.shape == (32, 64)
    # A neuron's charge is as large at the start with twice the synapses.
    spread = [model.state_reads["0"].tag.weight.std() * math.sqrt(model.state_reads["0"].tag.weight.shape[1])
              for model in (plain, late)]
    assert abs(spread[0] / spread[1] - 1) < 0.1
    tokens = _bytes()
    assert not torch.equal(plain(tokens), late(tokens)) and not torch.equal(late(tokens), both(tokens))


def test_every_channel_comes_once_more_after_its_own_delay():
    model, tokens = _stack(), _bytes()
    read, spikes = model.state_reads["0"], _stream(model, tokens)
    heard = read.heard(spikes)
    assert heard.shape == (2, 80, 128) and torch.equal(heard[..., :64], spikes)
    # The late fibres follow the stream's own, delay by delay: channel c is late by 1 + c % 3 bytes.
    place = 64
    for delay in (1, 2, 3):
        channels = torch.arange(delay - 1, 64, 3)
        fibres = heard[..., place:place + len(channels)]
        assert torch.equal(fibres[:, delay:], spikes[:, :-delay][..., channels])
        # A fibre is silent until its first byte has arrived.
        assert not fibres[:, :delay].any()
        place += len(channels)
    assert place == 128
    assert torch.equal(_stack(read_delays=1).state_reads["0"].heard(spikes), spikes)


def test_a_tag_tells_what_led_up_to_its_cell():
    plain, late, tokens = _stack(read_delays=1), _stack(), _bytes()
    spikes = _stream(late, tokens)
    changed = spikes.clone()
    changed[:, 40] = spikes[:, 41]
    for model, reach in ((plain, 0), (late, 3)):
        read = model.state_reads["0"]
        probe, tag, payload = read.packets(spikes)
        other_probe, other_tag, other_payload = read.packets(changed)
        moved = (tag != other_tag).any(-1).any(1).any(0).nonzero().flatten().tolist()
        # The tags of the cell and of the cells up to three bytes after it hear the change.
        assert moved == list(range(40, 41 + reach))
        # The probes and the payloads hear the stream as it is.
        assert (probe != other_probe).any(-1).any(1).any(0).nonzero().flatten().tolist() == [40]
        assert (payload != other_payload).any(-1).any(1).any(0).nonzero().flatten().tolist() == [40]
    read = _stack(read_delay_probes=True).state_reads["0"]
    moved = (read.packets(spikes)[0] != read.packets(changed)[0]).any(-1).any(1).any(0).nonzero().flatten().tolist()
    assert moved == [40, 41, 42, 43]


def test_one_read_finds_the_cell_that_followed_bytes_like_the_present_one():
    model = _stack(grown=False, read_delays=2, dim=64, read_width=64, read_heads=1, read_budget=64)
    read = model.state_reads["0"]
    with torch.no_grad():
        # The probe hears the stream as it is, the tag hears the same channels one byte late, and nothing else.
        read.probe.weight.copy_(torch.eye(64) * 0.5)
        read.tag.weight.copy_(torch.cat((torch.zeros(64, 64), torch.eye(64) * 0.5), 1))
        read.gain.fill_(math.log(8.0) / read.rate)
    # A text in which the present byte came once before.
    tokens = torch.tensor([[17, 201, 5, 88, 143, 30, 9, 250, 61, 5]])
    shares = read.shares(_stream(model, tokens), tokens)[0]
    # The last byte's read turns to the cell after the earlier 5: the byte that followed it.
    assert shares[0, 0, -1].argmax() == 3 and shares[0, 0, -1, 3] > 0.5


def test_a_stack_with_late_fibres_fires_whole_spikes_and_sees_no_later_byte():
    for probes in (False, True):
        model, tokens = _stack(read_delay_probes=probes), _bytes()
        changed = tokens.clone()
        changed[:, 50:] = _bytes(seed=7)[:, 50:]
        first, second = model(tokens), model(changed)
        assert torch.equal(first, first.round())
        assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
        for packet in model.state_reads["0"].packets(_stream(model, tokens)):
            assert torch.equal(packet, packet.round())


def test_every_synapse_of_a_late_fibre_learns():
    for mode in ("race", "soft"):
        model = _stack(read_mode=mode, read_delay_probes=True, **({} if mode == "race" else dict(read_budget=None)))
        model(_bytes()).float().square().mean().backward()
        for read in model.state_reads.values():
            for synapse in (read.probe, read.tag):
                late = synapse.weight.grad[:, 64:]
                assert late.shape == (32, 64) and late.abs().sum(0).gt(0).all(), mode


def test_the_exact_sums_count_the_late_fibres():
    model = _stack(read_delay_probes=True)
    model._check_accumulator()
    read = model.state_reads["1"]
    bounds = read.accumulator_bounds()
    for index, synapse in enumerate((read.probe, read.tag)):
        # A neuron sums the synapses of both of its fibres from every channel.
        assert synapse.weight.shape[1] == 128
        assert bounds[index] == synapse.weight_ticks().abs().sum(-1).amax()
    with torch.no_grad():
        read.tag.weight.fill_(1.0)
    limit = int(bounds[1]) // 2
    assert read.scale_synapses(limit) >= 32
    # Rounding moves each of a neuron's 128 synapses by half a tick at most.
    assert read.accumulator_bounds()[1] <= limit


@pytest.mark.parametrize("setting", [dict(read_delays=0), dict(read_delays=17), dict(read_delays=2.0),
                                     dict(read_delays=1, read_delay_probes=True),
                                     dict(read_mode="age", read_budget=None)])
def test_a_read_refuses_late_fibres_it_cannot_have(setting):
    with pytest.raises(ValueError, match="late fibres|delays"):
        _stack(**setting)
