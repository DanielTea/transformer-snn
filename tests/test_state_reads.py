import pytest
import torch
from torch.nn import functional as F

from snnformer.event_lm import EventLM, EventLMConfig


def _decoder(seed=0, **settings):
    base = dict(dim=96, hidden=40, depth=6, route="steered_correction_code", carry=False, memory_heads=4,
                memory_slots=14, resolution=16, memory_width=32, forecast_key_channels=2, match_forecast_key=True,
                learn_context_keys=True, compact_delays=True, compact_axon_state=True, activity_inhibition=True,
                inhibition_by_expansion=True, local_trace=True, trace_write_credit=True, router_charge_bits=8,
                residual_correction_start=3, compact_trace_state=True, delay_group_widths=(12, 12, 8, 8),
                trace_arrival="selected", trace_arrival_groups=2, clock_start=4, clock_threshold=8)
    torch.manual_seed(seed)
    return EventLM(EventLMConfig(**dict(base, **settings)))


def _reads(mode, **settings):
    return _decoder(read_after=(1, 3), read_width=32, read_heads=4, read_window=24, read_mode=mode, **settings)


def _grow(model):
    """Give the silent reads output weights and age offsets."""
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
            read.age.copy_(torch.randn(read.age.shape, generator=generator) * 0.5)
    return model


@pytest.mark.parametrize("mode", ["soft", "race", "age"])
def test_untrained_reads_leave_the_decoder_as_it_is(mode):
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    plain, model = _decoder(), _reads(mode)
    # The reads are created last and their output synapses start at zero.
    assert torch.equal(model(tokens), plain(tokens))
    assert not torch.equal(_grow(model)(tokens), plain(tokens))


@pytest.mark.parametrize("mode", ["soft", "race", "age"])
def test_reads_see_only_earlier_cells_within_the_window(mode):
    model = _grow(_reads(mode))
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    later = tokens.clone()
    later[:, 50:] = (later[:, 50:] + 7) % 256
    first, second = model(tokens), model(later)
    assert torch.equal(first[:, :50], second[:, :50])
    assert not torch.equal(first[:, 50:], second[:, 50:])
    read = model.state_reads["1"]
    spikes = torch.randint(-20, 21, (2, 40, 96), generator=torch.Generator().manual_seed(2)).float()
    shares, payload = read.shares(spikes)
    distance = torch.arange(40)[:, None] - torch.arange(40)
    assert shares[..., (distance < 0) | (distance >= 24)].abs().sum() == 0
    assert torch.allclose(shares.sum(-1), torch.ones(()))
    assert payload.shape == (2, 4, 40, 8)
    # Without an address the shares are the same for every stream.
    assert shares.shape[0] == (1 if mode == "age" else 2)


def test_race_reads_pass_whole_pulses():
    model = _grow(_reads("race"))
    read = model.state_reads["3"]
    spikes = torch.randint(-20, 21, (2, 40, 96), generator=torch.Generator().manual_seed(2)).float()
    shares, payload = read.shares(spikes)
    assert torch.equal(payload, payload.round())
    # Every cell fires a whole number of spikes up to the limit, and the most charged cell fires all of them.
    fired = shares * read.levels / shares.amax(-1, keepdim=True)
    assert torch.allclose(fired, fired.round(), atol=1e-4)
    assert fired.max() == read.levels
    assert torch.equal(read(spikes), read(spikes).round())


def test_a_race_of_rates_spends_its_budget_as_a_softmax_in_whole_spikes():
    spikes = torch.randint(-20, 21, (2, 40, 96), generator=torch.Generator().manual_seed(2)).float()
    coarse, fine, exact = (_grow(_reads("race", read_budget=budget)).state_reads["1"]
                           for budget in (64, 4096, 1 << 24))
    pulses = []
    assert torch.equal(coarse(spikes, pulses), coarse(spikes).round())
    # Each head's race ends near its budget of whole spikes.
    assert pulses[0] == pulses[0].round() and abs(pulses[0] / (2 * 4 * 40) - 64) < 4
    # Equal seeds give the reads equal synapses, so their shares differ only by the rounding to whole spikes.
    difference = (fine.shares(spikes)[0] - exact.shares(spikes)[0]).abs().max()
    assert difference < 0.01 < (coarse.shares(spikes)[0] - exact.shares(spikes)[0]).abs().max()


def test_race_reads_count_their_cell_spikes_and_read_pulses():
    spikes = torch.randint(-20, 21, (2, 40, 96), generator=torch.Generator().manual_seed(2)).float()
    for mode, entries in (("soft", 0), ("race", 2)):
        pulses = []
        _grow(_reads(mode)).state_reads["3"](spikes, pulses)
        assert len(pulses) == entries
    cells, read_pulses = pulses
    assert cells > 0 and cells == cells.round() and read_pulses > 0 and read_pulses == read_pulses.round()
    # A coarser read counts fewer pulses for the same cells.
    coarse = []
    _grow(_reads("race", read_resolution=2)).state_reads["3"](spikes, coarse)
    assert coarse[0] == cells and coarse[1] < read_pulses
    # Synapses and thresholds that are larger by the same factor fire nearly the same pulses.
    for name in ("probe", "payload"):
        scaled = _reads("race", read_synapse_gain=4).state_reads["3"]
        plain = _reads("race").state_reads["3"]
        getattr(scaled, name).weight.data.copy_(getattr(plain, name).weight * 4)
        quanta = torch.full((2, 40, 1), 5.0)
        drive = getattr(plain, name)(spikes)
        difference = scaled.fire(getattr(scaled, name)(spikes), quanta * 4) - plain.fire(drive, quanta)
        assert difference.abs().max() <= 1 and difference.abs().mean() < 0.1


@pytest.mark.parametrize("mode,settings", [("soft", {}), ("race", {}), ("race", {"read_signed": False}),
                                           ("race", {"read_budget": 64}), ("age", {})])
def test_reads_train_through_their_address_and_payload(mode, settings):
    model = _grow(_reads(mode, **settings)).train()
    tokens = torch.randint(0, 256, (2, 64), generator=torch.Generator().manual_seed(1))
    F.cross_entropy(model(tokens).flatten(0, 1) / 16, tokens.flatten()).backward()
    for name, parameter in model.state_reads.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_the_integer_guard_covers_spike_reads_and_not_the_softmax_reference():
    model = _grow(_reads("race", read_budget=16))
    model._check_accumulator()
    with torch.no_grad():
        model.state_reads["1"].payload.weight.fill_(1.9)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        model._check_accumulator()
    # A coarser grain of the read pulses lets more of them reach a read neuron.
    model = _grow(_reads("race", read_budget=16))
    with torch.no_grad():
        model.state_reads["3"].grain.fill_(12.0)
    with pytest.raises(ValueError, match="exact FP32 accumulator"):
        model._check_accumulator()
    soft = _grow(_reads("soft"))
    with torch.no_grad():
        soft.state_reads["1"].payload.weight.fill_(1.9)
    soft._check_accumulator()


def test_reads_follow_byte_rate_blocks_and_have_no_stream_state_yet():
    with pytest.raises(ValueError, match="byte-rate blocks"):
        _decoder(read_after=(4,))
    with pytest.raises(ValueError, match="byte-rate blocks"):
        _decoder(read_after=(1, 1))
    with pytest.raises(ValueError, match="heads divide"):
        _decoder(read_after=(1,), read_width=30)
    with pytest.raises(ValueError, match="positive sizes"):
        _decoder(read_after=(1,), read_resolution=0)
    with pytest.raises(ValueError, match="positive sizes"):
        _decoder(read_after=(1,), read_mode="race", read_synapse_gain=1.5)
    model = _reads("soft")
    with pytest.raises(ValueError, match="no carried stream state"):
        model.forward_chunk(torch.zeros(1, 8, dtype=torch.long))


def _text(line):
    return torch.tensor([list(line.encode())])


@pytest.mark.parametrize("mode", ["soft", "race", "age"])
def test_word_ages_leave_a_read_as_it_is_until_they_are_learned(mode):
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    plain, aged = _grow(_reads(mode)), _grow(_reads(mode, read_beats=3))
    assert torch.equal(aged(tokens), plain(tokens))
    with torch.no_grad():
        for read in aged.state_reads.values():
            read.beat_age[:, 2] = 4.0
    assert not torch.equal(aged(tokens), plain(tokens))


def test_a_cell_counts_the_words_that_started_after_it():
    read = _reads("race", read_beats=2).state_reads["1"]
    #                  a  b  _  c  d  ,  _  e  f  _  _  g
    started = torch.tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4])
    ages = read.word_ages(_text("ab cd, ef  g"))[0]
    assert torch.equal(ages, (started[:, None] - started).clamp(0, 2))
    # A digit and a byte of a longer character belong to words, and a word at the first byte starts there.
    assert read.word_ages(_text("7é x"))[0, -1].tolist() == [1, 1, 1, 1, 0]


def test_a_head_can_prefer_the_word_before_last():
    read = _reads("age", read_beats=3).state_reads["1"]
    with torch.no_grad():
        read.beat_age[:, 2] = 30.0
    tokens = _text("one two three four five six")
    spikes = torch.randint(-20, 21, (1, tokens.shape[1], 96), generator=torch.Generator().manual_seed(2)).float()
    shares, _ = read.shares(spikes, tokens)
    # At the last byte the word before last is "four " at bytes 14 to 18, whatever the lengths of the words.
    assert shares[0, :, -1, 14:19].sum(-1).min() > 0.999
    with pytest.raises(ValueError, match="reads the bytes"):
        read.shares(spikes)


def test_word_ages_use_only_the_bytes_so_far():
    model = _grow(_reads("race", read_beats=3))
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.beat_age.copy_(torch.randn(read.beat_age.shape, generator=generator))
    letters = torch.tensor(list(b"abcde   ,."))
    tokens = letters[torch.randint(0, len(letters), (2, 80), generator=generator)]
    later = tokens.clone()
    later[:, 50:] = letters[torch.randint(0, len(letters), (2, 30), generator=generator)]
    first, second = model(tokens), model(later)
    assert torch.equal(first[:, :50], second[:, :50])
    assert not torch.equal(first[:, 50:], second[:, 50:])
    with pytest.raises(ValueError, match="up to sixteen"):
        _reads("race", read_beats=17)
