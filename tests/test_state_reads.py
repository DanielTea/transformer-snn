import itertools

import pytest
import torch
from torch.nn import functional as F

from snnformer.event_lm import EventLM, EventLMConfig, _beat_age_chances


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


def test_a_clock_at_word_ends_gives_a_word_one_age_from_every_later_byte():
    read = _reads("race", read_beats=2, read_clock="end").state_reads["1"]
    #                 a  b  _  c  d  ,  _  e  f  _  _  g
    ended = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3])
    tokens = _text("ab cd, ef  g")
    ages = read.word_ages(tokens)[0]
    assert torch.equal(ages, (ended[:, None] - ended).clamp(0, 2))
    # Seen from the byte after "cd", from the gap and from inside "ef", the cells of "cd" have the age one.
    assert ages[5:9, 3:5].unique().tolist() == [1]
    # With the clock at word starts the same cells have the age nought from the gap and one from inside "ef".
    starts = _reads("race", read_beats=2).state_reads["1"].word_ages(tokens)[0]
    assert starts[5:7, 3:5].unique().tolist() == [0] and starts[7:9, 3:5].unique().tolist() == [1]
    with pytest.raises(ValueError, match="need such ages"):
        _reads("race", read_clock="end")


def test_a_beat_marks_its_cell_so_a_head_can_address_the_end_of_a_word():
    read = _reads("age", read_beats=3, read_clock="end", read_marks=True).state_reads["1"]
    tokens = _text("one two three four five six")
    spikes = torch.randint(-20, 21, (1, tokens.shape[1], 96), generator=torch.Generator().manual_seed(2)).float()
    # The marks start silent.
    unmarked = _reads("age", read_beats=3, read_clock="end").state_reads["1"]
    assert torch.equal(read.shares(spikes, tokens)[0], unmarked.shares(spikes, tokens)[0])
    with torch.no_grad():
        read.beat_mark[:, 1] = 30.0
    shares, _ = read.shares(spikes, tokens)
    # The last byte lies in "six". The first bytes after "five" and "four" are the marked cells of the ages
    # nought and one, at bytes 23 and 18, and one offset picks the single cell after "four".
    assert shares[0, :, -1, 18].min() > 0.999
    with torch.no_grad():
        read.beat_mark[:, 1] = 0.0
        read.beat_age[:, 1] = 30.0
    # The offset for the age alone spreads over every cell of that age: the space after "four" and "five" itself.
    shares, _ = read.shares(spikes, tokens)
    assert shares[0, :, -1, 18:23].sum(-1).min() > 0.999 and shares[0, :, -1, 18].max() < 0.25
    with pytest.raises(ValueError, match="marks its cell only"):
        _reads("race", read_marks=True)


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


@pytest.mark.parametrize("settings", [{}, {"read_clock": "end"}, {"read_clock": "end", "read_marks": True}])
def test_word_ages_use_only_the_bytes_so_far(settings):
    model = _grow(_reads("race", read_beats=3, **settings))
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.beat_age.copy_(torch.randn(read.beat_age.shape, generator=generator))
            if read.beat_mark is not None:
                read.beat_mark.copy_(torch.randn(read.beat_mark.shape, generator=generator))
    letters = torch.tensor(list(b"abcde   ,."))
    tokens = letters[torch.randint(0, len(letters), (2, 80), generator=generator)]
    later = tokens.clone()
    later[:, 50:] = letters[torch.randint(0, len(letters), (2, 30), generator=generator)]
    first, second = model(tokens), model(later)
    assert torch.equal(first[:, :50], second[:, :50])
    assert not torch.equal(first[:, 50:], second[:, 50:])
    with pytest.raises(ValueError, match="up to sixteen"):
        _reads("race", read_beats=17)


def test_heads_can_start_with_an_age_in_words_of_their_own():
    read = _reads("race", read_beats=2, read_beat_start=1.5).state_reads["1"]
    # Four heads and three ages: the fourth head starts where the first does.
    assert read.beat_age.tolist() == [[1.5, 0, 0], [0, 1.5, 0], [0, 0, 1.5], [1.5, 0, 0]]
    with pytest.raises(ValueError, match="needs such ages"):
        _reads("race", read_beat_start=1.5)


def test_the_chances_of_beat_ages_are_those_over_all_clocks():
    rates = torch.tensor([0.2, 0.9, 0.5, 0.0, 0.7, 0.3], dtype=torch.float64)
    chances = _beat_age_chances(rates, 4)
    counted = torch.zeros(4, 6, 6, dtype=torch.float64)
    for pattern in itertools.product((0, 1), repeat=6):
        beats = torch.tensor(pattern)
        chance = torch.where(beats.bool(), rates, 1 - rates).prod()
        started = beats.cumsum(0)
        ages = started[:, None] - started
        for age in range(4):
            counted[age] += chance * (ages == age)
    for age in range(4):
        assert torch.allclose(chances[age], counted[age].tril(), atol=1e-9)


def _clocked(mode, **settings):
    return _reads(mode, read_beats=3, read_clock="learned", **settings)


def test_the_clock_cells_of_a_read_start_deaf_and_beat_at_their_own_pace():
    read = _clocked("race").state_reads["1"]
    spikes = torch.randint(-20, 21, (2, 400, 96), generator=torch.Generator().manual_seed(2)).float()
    beats, rates = read.clock_beats(spikes)
    assert beats.shape == (2, 400, 4) and set(beats.unique().tolist()) == {0.0, 1.0}
    # A quarter of the bytes, as near as a resting charge in quarter spikes comes to it.
    assert torch.allclose(rates, torch.full_like(rates, torch.tensor(-1.0).sigmoid().item()))
    assert (beats.mean((0, 1)) - rates.mean((0, 1))).abs().max() < 0.01
    # The heads' clock cells beat at different bytes.
    assert not torch.equal(beats[..., 0], beats[..., 1])
    with torch.no_grad():
        assert read.clock_beats(spikes)[1] is None
    with pytest.raises(ValueError, match="need such ages"):
        _reads("race", read_clock="learned")
    with pytest.raises(ValueError, match="need such ages"):
        _clocked("race", read_clock_rate=1.0)


def test_a_head_counts_a_cells_age_in_the_beats_of_its_clock_cell():
    read = _clocked("race").state_reads["1"]
    tokens = _text("one two three four five six")
    starts = [0, 4, 8, 14, 19, 24]
    # Channel 0 of the stream marks the word starts, one tag passes it on, and every clock cell hears only that
    # tag. No probe meets a tag, so the ages alone decide the shares.
    spikes = torch.zeros(1, tokens.shape[1], 96)
    spikes[0, starts, 0] = 320.0
    with torch.no_grad():
        read.probe.weight.zero_()
        read.tag.weight.zero_()
        read.tag.weight[0, 0] = 1.0
        read.clock.weight.zero_()
        read.clock.weight[:, 0] = 1.9
        read.clock_rest.fill_(8.0)
        read.beat_age[:, 2] = 30.0
    beats, _ = read.clock_beats(spikes)
    assert beats[0].nonzero()[:, 0].unique().tolist() == starts and beats[0, starts].min() == 1
    # The beats are the same where the read has made its tags already.
    assert torch.equal(read.clock_beats(spikes, read.packets(spikes)[1])[0], beats)
    shares, _ = read.shares(spikes)
    # At the last byte the word before last is "four " at bytes 14 to 18, as with the word clock and no bytes read.
    assert shares[0, :, -1, 14:19].sum(-1).min() > 0.999
    with pytest.raises(ValueError, match="hears the tags"):
        _clocked("age")


def test_the_synapses_of_a_clock_cell_cannot_leave_the_range_of_exact_sums():
    # With 256 tags a synapse at the usual limit of 508 ticks would allow sums of 130,048 ticks, where 32,768
    # times the largest count of 512 pulses is the last exact one.
    model = _decoder(read_after=(1, 3), read_width=256, read_heads=4, read_window=24, read_mode="race",
                     read_beats=3, read_clock="learned")
    for read in model.state_reads.values():
        with torch.no_grad():
            read.project.weight.normal_(std=0.01, generator=torch.Generator().manual_seed(3))
            read.clock.weight.fill_(40.0)
            read.clock.weight[:, ::2] = -40.0
            read.clock_rest.fill_(-40.0)
        assert read.clock_limit == 127 and read.clock_ticks().abs().unique().tolist() == [127]
    model._check_accumulator()
    tokens = torch.randint(0, 256, (2, 80), generator=torch.Generator().manual_seed(1))
    assert torch.isfinite(model(tokens)).all()
    # A narrow read keeps the limit of every other synapse.
    assert _clocked("race").state_reads["1"].clock_limit == 508


@pytest.mark.parametrize("marks", [False, True])
def test_learned_read_clocks_use_only_the_stream_so_far(marks):
    model = _grow(_clocked("race", read_marks=marks))
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.beat_age.copy_(torch.randn(read.beat_age.shape, generator=generator))
            read.clock.weight.copy_(torch.randn(read.clock.weight.shape, generator=generator) * 0.3)
            if marks:
                read.beat_mark.copy_(torch.randn(read.beat_mark.shape, generator=generator))
    tokens = torch.randint(0, 256, (2, 80), generator=generator)
    later = tokens.clone()
    later[:, 50:] = torch.randint(0, 256, (2, 30), generator=generator)
    first, second = model(tokens), model(later)
    assert torch.equal(first[:, :50], second[:, :50])
    assert not torch.equal(first[:, 50:], second[:, 50:])
    model._check_accumulator()


@pytest.mark.parametrize("mode,marks", [("soft", False), ("race", False), ("race", True)])
def test_the_clock_cells_of_a_read_learn_from_the_mean_offsets_over_their_chances(mode, marks):
    model = _grow(_clocked(mode, read_beat_start=1.0, read_marks=marks)).train()
    if marks:
        with torch.no_grad():
            for read in model.state_reads.values():
                read.beat_mark.normal_(std=0.5, generator=torch.Generator().manual_seed(4))
    tokens = torch.randint(0, 256, (2, 64), generator=torch.Generator().manual_seed(1))
    F.cross_entropy(model(tokens).flatten(0, 1) / 16, tokens.flatten()).backward()
    for name, parameter in model.state_reads.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    # The whole beats decide the output: the mean offsets add a slope and nothing else.
    read = model.state_reads["1"]
    spikes = torch.randint(-20, 21, (2, 40, 96), generator=torch.Generator().manual_seed(2)).float()
    with torch.no_grad():
        scored = read(spikes)
    assert torch.equal(read(spikes), scored)
