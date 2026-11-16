from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from snnformer.event_lm import EventLM, EventLMConfig, _clock_ticks, _unmatched_winner_spikes


_PRIOR = EventLMConfig(dim=32, hidden=16, depth=2, route="steered_correction_code", carry=False,
                       compact_delays=True, compact_axon_state=True, compact_trace_state=True, local_trace=True,
                       activity_inhibition=True, resolution=4, trace_arrival="selected", trace_arrival_groups=2)


def _grown(threshold, source="surprise", leak=0, seed=711):
    torch.manual_seed(seed)
    parent = EventLM(_PRIOR)
    cfg = replace(_PRIOR, depth=5, residual_correction_start=2, clock_start=2, clock_threshold=threshold,
                  clock_leak=leak, clock_source=source)
    model = EventLM(cfg).eval()
    model.initialize_deeper(_PRIOR, parent.state_dict())
    return parent, model


def _activate(model):
    with torch.no_grad():
        for block in model.blocks[model.cfg.clock_start:]:
            block.expand.weight.normal_(std=0.25)


def test_clock_neuron_fires_in_proportion_to_surprise_and_never_falls_a_threshold_behind():
    generator = torch.Generator().manual_seed(5)
    surprise = torch.randint(0, 40, (4, 300), generator=generator)
    ticks, membrane = _clock_ticks(surprise, 40, 0)
    total = surprise.cumsum(1)
    torch.testing.assert_close(ticks.cumsum(1), total // 40, rtol=0, atol=0)
    torch.testing.assert_close(membrane.long(), total[:, -1] % 40, rtol=0, atol=0)
    # The second half resumes from the carried membrane.
    first, carried = _clock_ticks(surprise[:, :111], 40, 0)
    second, final = _clock_ticks(surprise[:, 111:], 40, 0, carried)
    assert torch.equal(torch.cat((first, second), dim=1), ticks) and torch.equal(final, membrane)


def test_clock_input_saturates_and_leak_holds_the_membrane_at_rest():
    ticks, membrane = _clock_ticks(torch.tensor([[900, 0, 0, 5, 5, 5, 5, 30, 0]]), 30, 10)
    assert ticks.tolist() == [[False, False, False, False, False, False, False, False, False]]
    assert membrane.tolist() == [10]
    ticks, membrane = _clock_ticks(torch.tensor([[900, 900, 25, 25]]), 30, 0)
    assert ticks.tolist() == [[True, True, False, True]] and membrane.tolist() == [20]


def test_surprise_counts_winner_spikes_that_the_arriving_byte_did_not_match():
    counts = torch.zeros(1, 3, 256, dtype=torch.int32)
    counts[0, 0, 7], counts[0, 0, 9] = 50, 20
    counts[0, 1, 4] = 11
    tokens = torch.tensor([[1, 9, 4]])
    assert _unmatched_winner_spikes(counts, tokens).tolist() == [[0, 30, 0]]
    previous = torch.zeros(1, 256, dtype=torch.int32)
    previous[0, 3] = 6
    assert _unmatched_winner_spikes(counts, tokens, previous).tolist() == [[6, 30, 0]]


@pytest.mark.parametrize(("source", "threshold", "leak"), [("surprise", 24, 0), ("surprise", 38, 5),
                                                            ("periodic", 3, 0)])
def test_clocked_growth_is_silent_and_matches_native_steps_and_uneven_chunks(source, threshold, leak):
    parent, model = _grown(threshold, source, leak)
    tokens = torch.randint(256, (3, 61), generator=torch.Generator().manual_seed(8))
    native = model.to_integer()
    torch.testing.assert_close(native(tokens), parent.to_integer()(tokens), rtol=0, atol=0)
    _activate(model)
    native = model.to_integer()
    expected = native(tokens)
    assert not torch.equal(expected, parent.to_integer()(tokens))
    torch.testing.assert_close(model(tokens).to(torch.int32), expected, rtol=0, atol=0)
    state = None
    for index, byte in enumerate(tokens.unbind(1)):
        counts, state = native.step(byte, state)
        torch.testing.assert_close(counts, expected[:, index], rtol=0, atol=0)
    events = state.clock.events
    assert 0 < events.min() and events.max() < tokens.shape[1]
    if source == "surprise":
        assert events.unique().numel() > 1
    assert state.clock.membrane.dtype == torch.int16 and state.clock.held.dtype == torch.int16
    assert state.clock.forecast.dtype == torch.int8 and state.clock.forecast.abs().max() <= 127
    assert 0 <= state.clock.membrane.min() and state.clock.membrane.max() < threshold
    for sizes in ((1, 3, 13, 35, 9), (7,) * 8 + (5,)):
        chunk, cursor = None, 0
        for part in tokens.split(sizes, dim=1):
            scores, chunk = native.forward_chunk(part, chunk)
            torch.testing.assert_close(scores, expected[:, cursor:cursor + part.shape[1]], rtol=0, atol=0)
            cursor += part.shape[1]
        assert torch.equal(chunk.clock.events, events)
        torch.testing.assert_close(chunk.clock.held.to(torch.int16), state.clock.held, rtol=0, atol=0)
        torch.testing.assert_close(chunk.clock.membrane.to(torch.int16), state.clock.membrane, rtol=0, atol=0)
        for streamed, batched in zip(state.layers, chunk.local_layers, strict=True):
            torch.testing.assert_close(streamed.trace_charge, batched.trace_charge, rtol=0, atol=0)
            torch.testing.assert_close(streamed.history.long().sum(), batched.history.long().sum(), rtol=0, atol=0)


def test_a_tick_on_every_byte_reproduces_the_unclocked_blocks():
    _, model = _grown(1, "periodic")
    _activate(model)
    ordinary = EventLM(replace(model.cfg, clock_start=None, clock_threshold=0, clock_source="surprise")).eval()
    ordinary.load_state_dict(model.state_dict())
    tokens = torch.randint(256, (2, 47), generator=torch.Generator().manual_seed(3))
    torch.testing.assert_close(model.to_integer()(tokens), ordinary.to_integer()(tokens), rtol=0, atol=0)


def test_clocked_delays_read_earlier_events_and_hold_between_ticks():
    _, model = _grown(4, "periodic")
    _activate(model)
    native = model.to_integer()
    tokens = torch.randint(256, (1, 40), generator=torch.Generator().manual_seed(11))
    state, held = None, []
    for byte in tokens.unbind(1):
        _, state = native.step(byte, state)
        held.append(state.clock.held.clone())
    held = torch.stack(held, dim=1)
    # Ticks fall on bytes 3, 7, 11, ... and the correction is constant in between.
    assert state.clock.events.tolist() == [10]
    assert not held[:, :3].any()
    for tick in range(3, 36, 4):
        assert all(torch.equal(held[:, tick], held[:, tick + offset]) for offset in (1, 2, 3))
    assert any(not torch.equal(held[:, tick], held[:, tick + 4]) for tick in range(3, 36, 4))
    # A byte between ticks never reaches the clocked blocks.
    changed = tokens.clone()
    changed[0, 9] = (changed[0, 9] + 1) % 256
    state, other = None, []
    for byte in changed.unbind(1):
        _, state = native.step(byte, state)
        other.append(state.clock.held.clone())
    assert torch.equal(torch.stack(other, dim=1)[:, :11], held[:, :11])


def test_predictions_before_a_change_ignore_later_bytes_and_other_streams():
    _, model = _grown(24)
    _activate(model)
    native = model.to_integer()
    tokens = torch.randint(256, (3, 61), generator=torch.Generator().manual_seed(8))
    expected = native(tokens)
    changed = tokens.clone()
    changed[:, 40:] = torch.randint(256, (3, 21), generator=torch.Generator().manual_seed(9))
    changed[1] = torch.randint(256, (61,), generator=torch.Generator().manual_seed(10))
    actual = native(changed)
    torch.testing.assert_close(actual[[0, 2], :40], expected[[0, 2], :40], rtol=0, atol=0)


def test_training_reaches_clocked_and_byte_rate_synapses_and_returns_the_forecast():
    _, model = _grown(24)
    _activate(model)
    model.train()
    tokens = torch.randint(256, (3, 61), generator=torch.Generator().manual_seed(8))
    counts, forecast = model(tokens[:, :-1], early_readout=2)
    assert forecast.shape == counts.shape and model.forecast_resolution == 1
    # The forecast population fires once for every four output spikes of the same current.
    assert forecast.abs().sum() < counts.abs().sum() / 2
    loss = F.cross_entropy(counts.flatten(0, 1) / 8, tokens[:, 1:].flatten())
    loss.backward()
    assert all(parameter.grad is not None and parameter.grad.isfinite().all() and parameter.grad.abs().sum() > 0
               for parameter in model.parameters())
    with pytest.raises(ValueError, match="early readout"):
        model(tokens, early_readout=1)


def test_muting_the_clock_returns_the_byte_rate_prediction_and_displacing_it_changes_the_output():
    parent, model = _grown(24)
    _activate(model)
    tokens = torch.randint(256, (2, 61), generator=torch.Generator().manual_seed(8))
    native = model.to_integer()
    expected = native(tokens)
    native.clock_intervention = "mute"
    torch.testing.assert_close(native(tokens), parent.to_integer()(tokens), rtol=0, atol=0)
    state, steps = None, []
    for byte in tokens.unbind(1):
        counts, state = native.step(byte, state)
        steps.append(counts)
    torch.testing.assert_close(torch.stack(steps, dim=1), parent.to_integer()(tokens), rtol=0, atol=0)
    native.clock_intervention = "shuffle"
    assert not torch.equal(native(tokens), expected)


@pytest.mark.parametrize(("settings", "message"), [
    ({"clock_threshold": 5}, "first clocked block"),
    ({"clock_start": 0, "clock_threshold": 5}, "byte-rate blocks below"),
    ({"clock_start": 1, "clock_threshold": 0}, "positive threshold"),
    ({"clock_start": 1, "clock_threshold": 5, "clock_leak": 5}, "positive threshold"),
    ({"clock_start": 1, "clock_threshold": 5, "clock_source": "words"}, "surprise or a constant drive"),
    ({"clock_start": 1, "clock_threshold": 5, "clock_source": "periodic", "clock_leak": 1}, "no leak"),
    ({"clock_start": 1, "clock_threshold": 5, "local_trace": False, "trace_arrival": None,
      "compact_trace_state": False}, "chunkable held charge"),
])
def test_invalid_clock_configuration(settings, message):
    with pytest.raises(ValueError, match=message):
        EventLM(replace(_PRIOR, **settings))


def test_depth_growth_keeps_trained_blocks_off_the_clock():
    parent = EventLM(_PRIOR)
    cfg = replace(_PRIOR, depth=4, residual_correction_start=2, clock_start=1, clock_threshold=9)
    with pytest.raises(ValueError, match="only the added blocks"):
        EventLM(cfg).initialize_deeper(_PRIOR, parent.state_dict())


def test_reported_stream_state_includes_the_clocked_blocks(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    from benchmark_decode import state_bytes
    from eval_event_lm import _stream_state_bytes

    _, model = _grown(24)
    _activate(model)
    native = model.to_integer()
    data = torch.randint(256, (200,), generator=torch.Generator().manual_seed(3))
    state = None
    sizes = []
    for byte in data:
        _, state = native.step(byte[None], state)
        sizes.append(state_bytes(state))
    # The clocked blocks hold no delay lines before the first tick.
    assert state.clock.events.item() > 0 and sizes[0] < sizes[-1]
    assert _stream_state_bytes(native, data) == sizes[-1]
    with pytest.raises(ValueError, match="did not fire"):
        _stream_state_bytes(native, data, limit=1)
