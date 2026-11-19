import math

import pytest
import torch

from snnformer.event_lm import _PHASE_SHORTEST, _PHASE_SPIKES, EventLM, EventLMConfig, _pacemaker_counts

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, read_phases=4, granule_after=_LEVELS, granule_clusters=1,
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


def test_pacemakers_fire_whole_spikes_in_fixed_rhythms():
    counts = _pacemaker_counts(200, 6, 96)
    assert counts.shape == (200, 12) and torch.equal(counts, counts.round())
    assert counts.abs().max() == _PHASE_SPIKES
    # A pair starts at the top of its cosine cell and at the rest of its sine cell.
    assert torch.equal(counts[0], torch.tensor([_PHASE_SPIKES] * 6 + [0] * 6, dtype=torch.float32))
    # The first pair turns once in the shortest period, the last once in the longest.
    assert _PHASE_SHORTEST == 3 and torch.equal(counts[::3, 0], torch.full((67,), float(_PHASE_SPIKES)))
    assert counts[48, 5] == -_PHASE_SPIKES and counts[96, 5] == _PHASE_SPIKES and counts[24, 11] == _PHASE_SPIKES
    # The periods run in equal ratios.
    turns = torch.tensor([3 * (96 / 3) ** (pair / 5) for pair in range(6)], dtype=torch.float64)
    exact = torch.cat(((2 * math.pi * 7 / turns).cos(), (2 * math.pi * 7 / turns).sin())) * _PHASE_SPIKES
    assert torch.equal(counts[7], exact.round().float())


def test_the_phases_at_two_bytes_tell_their_distance():
    counts = _pacemaker_counts(160, 8, 96).double()
    match = counts @ counts.T
    # The match of two bytes' phases depends on the bytes between them and not on where they are: every
    # diagonal of the matrix is flat, up to the rounding of the counts to whole spikes.
    slack = 16 * (_PHASE_SPIKES + 0.25)
    for distance in (0, 1, 2, 5, 17, 60):
        diagonal = match.diagonal(-distance)
        assert diagonal.max() - diagonal.min() <= 2 * slack
    # A byte matches itself best, and near bytes better than far ones.
    means = torch.stack([match.diagonal(-distance).mean() for distance in range(48)])
    assert means.argmax() == 0 and means[1] > means[6] > means[40]


def test_a_read_hears_the_pacemakers_only_when_asked():
    plain, heard = _stack(read_phases=0), _stack()
    extra = set(heard.state_dict()) - set(plain.state_dict())
    assert extra == {f"state_reads.{level}.{name}.weight" for level in _LEVELS for name in ("probe_phase", "tag_phase")}
    read = heard.state_reads["0"]
    assert read.probe_phase.weight.shape == read.tag_phase.weight.shape == (32, 8)
    assert not hasattr(plain.state_reads["0"], "probe_phase")
    # The pacemakers are no parameters and no part of a saved decoder.
    assert read.pacemakers.shape == (48, 8) and not any("pacemakers" in name for name in heard.state_dict())
    # A window longer than the read window gets the same rhythms, carried on.
    assert torch.equal(read.phase_counts(100)[:48], _pacemaker_counts(48, 4, 96))
    assert torch.equal(read.phase_counts(100), _pacemaker_counts(100, 4, 96))
    tokens = _bytes()
    assert not torch.equal(plain(tokens), heard(tokens))


def test_a_stack_with_pacemakers_fires_whole_spikes_and_sees_no_later_byte():
    model, tokens = _stack(), _bytes()
    changed = tokens.clone()
    changed[:, 50:] = _bytes(seed=7)[:, 50:]
    first, second = model(tokens), model(changed)
    assert torch.equal(first, first.round())
    assert torch.equal(first[:, :50], second[:, :50]) and not torch.equal(first[:, 50:], second[:, 50:])
    for name, packet in zip(("probe", "tag"), model.state_reads["0"].packets(
            model.input_emit(model.input(tokens) * 16))[:2], strict=True):
        assert torch.equal(packet, packet.round()), name


@pytest.mark.parametrize("distance", [1, 3, 7])
def test_the_phases_address_the_cell_a_chosen_number_of_bytes_back(distance):
    model, tokens = _stack(grown=False), _bytes(length=48)
    read = model.state_reads["0"]
    pairs = read.phases
    turns = 2 * math.pi * distance / (_PHASE_SHORTEST * (2 * read.window / _PHASE_SHORTEST) ** (
        torch.arange(pairs) / (pairs - 1)))
    with torch.no_grad():
        # No synapse from the stream: the heads address by the pacemakers alone.
        read.probe.weight.zero_()
        read.tag.weight.zero_()
        read.gain.fill_(math.log(4.0) / read.rate)
        # A tag neuron hears one pacemaker. A probe neuron hears the two cells of a pair in the mix that turns the
        # pair's phase back by the distance.
        tag = torch.eye(2 * pairs) * 0.25
        probe = torch.zeros(2 * pairs, 2 * pairs)
        for pair in range(pairs):
            cosine, sine = math.cos(turns[pair]), math.sin(turns[pair])
            probe[pair, pair], probe[pair, pairs + pair] = cosine, sine
            probe[pairs + pair, pair], probe[pairs + pair, pairs + pair] = -sine, cosine
        read.tag_phase.weight.copy_(tag.repeat(read.heads, 1))
        read.probe_phase.weight.copy_((probe * 0.25).repeat(read.heads, 1))
    shares = read.shares(model.input_emit(model.input(tokens) * 16), tokens)[0]
    # Every head of every byte puts its largest share on the cell that many bytes back.
    cells = torch.arange(distance, tokens.shape[1])
    assert torch.equal(shares[:, :, distance:].argmax(-1), (cells - distance).expand(2, read.heads, -1))
    assert shares[:, :, distance:].amax(-1).min() > 0.5


def test_every_synapse_from_the_pacemakers_learns():
    for mode in ("race", "soft"):
        model = _stack(read_mode=mode, **({} if mode == "race" else dict(read_budget=None)))
        model(_bytes()).float().square().mean().backward()
        for read in model.state_reads.values():
            for synapse in (read.probe_phase, read.tag_phase):
                assert synapse.weight.grad is not None and synapse.weight.grad.abs().sum() > 0, mode


def test_the_exact_sums_count_the_pacemakers():
    plain, heard = _stack(read_phases=0), _stack()
    heard._check_accumulator()
    for name, index in (("probe", 0), ("tag", 1)):
        read, bare = heard.state_reads["1"], plain.state_reads["1"]
        with torch.no_grad():
            getattr(bare, name).weight.copy_(getattr(read, name).weight)
            # The largest synapses a pacemaker can have.
            getattr(read, f"{name}_phase").weight.fill_(4.0)
        stream = getattr(read, name).weight_ticks().abs().sum(-1)
        # A pacemaker's 64 spikes weigh an eighth of a stream neuron's 512.
        assert _PHASE_SPIKES * 8 == read.emit.cap
        assert read.accumulator_bounds()[index] == (stream + 8 * 508 // 8).amax()
        assert bare.accumulator_bounds()[index] == stream.amax()
    heard._check_accumulator()


def test_the_pacemakers_pulses_are_counted_once():
    plain, heard, tokens = _stack(read_phases=0), _stack(), _bytes()
    without, reads = [], []
    plain(tokens, memory_pulses=without)
    heard(tokens, memory_pulses=reads)
    assert len(reads) == len(without) + 1 == 2 * len(_LEVELS) + 1
    assert reads[-1] == 2 * _pacemaker_counts(80, 4, 96).abs().sum()


@pytest.mark.parametrize("setting", [dict(read_phases=-1), dict(read_phases=65), dict(read_phases=2.0),
                                     dict(read_mode="age", read_budget=None)])
def test_a_read_refuses_pacemakers_it_cannot_hear(setting):
    with pytest.raises(ValueError, match="pacemaker"):
        _stack(**setting)
