from dataclasses import replace

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig
from snnformer.fast_synapses import FastSynapses


_BASE = EventLMConfig(dim=32, hidden=16, depth=2, route="steered_correction_code", carry=False,
                      compact_delays=True, compact_axon_state=True, compact_trace_state=True, local_trace=True,
                      activity_inhibition=True, resolution=4, trace_arrival="selected", trace_arrival_groups=2)
_SYNAPSES = dict(synapse_orders=(1, 2, 4), synapse_groups=3, synapse_cells=16, synapse_levels=2, synapse_gate=1)


def _memory(gate=2):
    memory = FastSynapses((2, 3, 5), groups=3, cells=8, levels=2, gate=gate)
    memory.readout[:, 1:] = torch.tensor([[3, 9], [4, 11], [5, 20]], dtype=torch.int32)
    return memory


def _streams(seed=3, batch=3, steps=260):
    generator = torch.Generator().manual_seed(seed)
    # A four-letter alphabet repeats contexts and forces key neurons to collide.
    return (torch.randint(0, 4, (batch, steps), generator=generator),
            torch.randint(0, 5, (batch, steps), generator=generator))


def _window(tokens, end, length):
    window = torch.zeros(length, dtype=torch.int64)
    kept = min(end, length)
    if kept:
        window[-kept:] = tokens[end - kept:end]
    return window[None]


def _reference_votes(memory, tokens, surprise):
    """One dictionary per head, updated byte by byte."""
    batch, steps = tokens.shape
    votes = torch.full((batch, steps, len(memory.orders), memory.groups), -1)
    for row in range(batch):
        tables = [{} for _ in memory.orders]
        for step in range(steps):
            before = memory._winners(_window(tokens[row], step, memory.longest))[0]
            after = memory._winners(_window(tokens[row], step + 1, memory.longest))[0]
            for head, order in enumerate(memory.orders):
                for cell in before[head].tolist() if step >= order and tokens[row, step] != 0 else ():
                    # Byte 0 is the mark of a key neuron without a synapse and is never stored.
                    if surprise[row, step] >= memory.gate:
                        tables[head][cell] = int(tokens[row, step])
                if step + 1 >= order:
                    votes[row, step, head] = torch.tensor([tables[head].get(cell, -1) for cell in after[head].tolist()])
    return votes


@pytest.mark.parametrize("gate", [0, 2, 4])
def test_each_key_neuron_points_at_the_last_byte_that_followed_it_with_error_spikes(gate):
    memory = _memory(gate)
    tokens, surprise = _streams()
    votes, levels, state = memory(tokens, surprise)
    expected = _reference_votes(memory, tokens, surprise)
    assert state is None and torch.equal(votes, expected)
    assert (votes >= 0).any() and (votes < 0).any()
    # A head stays silent until it has seen its whole context.
    assert (votes[:, 0, 1:] < 0).all() and (votes[:, 3, 2] < 0).all()


@pytest.mark.parametrize("sizes", [(260,), (1, 7, 100, 152), (52,) * 5, (3,) * 86 + (2,)])
def test_chunked_and_streaming_synapses_match_the_whole_stream(sizes):
    memory = _memory(4)
    tokens, surprise = _streams()
    whole, whole_levels, _ = memory(tokens, surprise)
    votes, levels, state = [], [], None
    for part, error in zip(tokens.split(sizes, dim=1), surprise.split(sizes, dim=1), strict=True):
        vote, level, state = memory(part, error, state, carry=True)
        votes.append(vote)
        levels.append(level)
    assert torch.equal(torch.cat(votes, dim=1), whole) and torch.equal(torch.cat(levels, dim=1), whole_levels)
    stream, stepped = memory.new_state(tokens.shape[0]), []
    for byte, error in zip(tokens.unbind(1), surprise.unbind(1), strict=True):
        vote, _, stream = memory.step(byte, error, stream)
        stepped.append(vote)
    assert torch.equal(torch.stack(stepped, dim=1), whole)
    assert torch.equal(stream.successor, state.successor) and torch.equal(stream.history, state.history)
    assert stream.successor.dtype == torch.uint8 and stream.seen == state.seen == memory.longest
    assert stream.written.any() and not stream.written.all()


def test_only_agreeing_synapses_fire_a_recall_neuron():
    memory = _memory()
    votes = torch.tensor([[[7, 7, 7], [7, 9, 9], [-1, 4, -1]]])
    levels = memory._levels(votes)
    # Three groups and two levels: two agreeing synapses reach level 1, three reach level 2.
    assert levels.tolist() == [[[2, 0, 0], [0, 1, 0], [0, 0, 0]]]
    spikes = memory.excitation(votes, levels)
    assert spikes[0, 7] == 9 and spikes[0, 9] == 4 and spikes.sum() == 13


def test_no_error_spikes_means_no_synapse_moves():
    memory = _memory(gate=1)
    tokens, _ = _streams()
    votes, _, state = memory(tokens, torch.zeros_like(tokens), carry=True)
    assert (votes < 0).all() and not state.written.any()


def test_a_synapse_keeps_its_byte_until_the_next_error_spike():
    memory = FastSynapses((1,), groups=1, cells=256, levels=1, gate=1)
    #                      a  b  a  c  a  d  a  e  a
    tokens = torch.tensor([[5, 6, 5, 7, 5, 8, 5, 9, 5]])
    errors = torch.tensor([[0, 0, 0, 1, 0, 0, 0, 1, 0]])
    # Only c and e arrive with error spikes after "a". The read after each "a" is at every second position.
    assert memory(tokens, errors)[0][0, ::2, 0, 0].tolist() == [-1, -1, 7, 7, 9]


def _decoder(clocked, seed=29):
    torch.manual_seed(seed)
    cfg = replace(_BASE, **_SYNAPSES)
    if clocked:
        cfg = replace(cfg, depth=4, residual_correction_start=2, clock_start=2, clock_threshold=24)
    model = EventLM(cfg).eval()
    with torch.no_grad():
        model.fast_synapses.readout[:, 1:] = torch.tensor([[2, 9], [5, 17], [8, 30]], dtype=torch.int32)
    return model


@pytest.mark.parametrize("clocked", [False, True])
def test_decoder_with_fast_synapses_agrees_in_float_integer_chunked_and_streaming_form(clocked):
    model = _decoder(clocked)
    native = model.to_integer()
    tokens = torch.randint(0, 4, (3, 97), generator=torch.Generator().manual_seed(8))
    expected = native(tokens)
    assert torch.equal(expected, model(tokens).to(torch.int32))
    native.fast_synapses.muted = True
    silent = native(tokens)
    native.fast_synapses.muted = False
    assert not torch.equal(silent, expected)
    plain = EventLM(replace(model.cfg, synapse_orders=None)).eval()
    plain.load_state_dict(model.state_dict(), strict=False)
    assert torch.equal(plain.to_integer()(tokens), silent)
    for sizes in ((97,), (1, 5, 40, 51), (13,) * 7 + (6,)):
        chunks, state = [], None
        for part in tokens.split(sizes, dim=1):
            counts, state = native.forward_chunk(part, state)
            chunks.append(counts)
        assert torch.equal(torch.cat(chunks, dim=1), expected)
    steps, stream = [], None
    for byte in tokens.unbind(1):
        counts, stream = native.step(byte, stream)
        steps.append(counts)
    assert torch.equal(torch.stack(steps, dim=1), expected)
    assert torch.equal(stream.synapses.successor, state.synapses.successor) and stream.synapses.written.any()
    # The clock already carries the forecast spikes. Without it the synapses keep their own copy.
    assert (stream.synapses.forecast is None) == clocked and (state.synapses.forecast is None) == clocked
    if not clocked:
        assert stream.synapses.forecast.dtype == torch.int8
        assert torch.equal(stream.synapses.forecast, state.synapses.forecast)


def test_recall_spikes_add_beyond_the_vocabulary_neurons_limit():
    native = _decoder(False).to_integer()
    cap = native.output_emit.cap
    with torch.no_grad():
        native.fast_synapses.readout[:, 1:] = 3 * cap
    tokens = torch.randint(0, 4, (2, 97), generator=torch.Generator().manual_seed(8))
    native.fast_synapses.capture = []
    counts = native(tokens)
    votes, levels, _ = native.fast_synapses.capture.pop()
    native.fast_synapses.muted = True
    silent = native(tokens)
    # The recall neurons are a population of their own and are counted with the vocabulary neurons.
    assert silent.abs().max() <= cap < counts.max()
    assert torch.equal(counts - silent, native.fast_synapses.excitation(votes, levels).to(counts.dtype))


def test_fast_synapses_see_only_the_past():
    native = _decoder(True).to_integer()
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, 4, (2, 80), generator=generator)
    other = tokens.clone()
    other[:, 50:] = torch.randint(0, 4, (2, 30), generator=generator)
    assert torch.equal(native(tokens)[:, :50], native(other)[:, :50])


def test_fast_synapses_reject_inconsistent_settings():
    with pytest.raises(ValueError, match="power-of-two"):
        EventLM(replace(_BASE, **{**_SYNAPSES, "synapse_cells": 12}))
    with pytest.raises(ValueError, match="levels within groups"):
        EventLM(replace(_BASE, **{**_SYNAPSES, "synapse_levels": 4}))
    with pytest.raises(ValueError, match="distinct positive context lengths"):
        EventLM(replace(_BASE, **{**_SYNAPSES, "synapse_orders": (2, 2)}))
    with pytest.raises(ValueError, match="chunkable held charge"):
        EventLM(replace(_BASE, **_SYNAPSES, local_trace=False, compact_trace_state=False, trace_arrival=None))
    model = EventLM(replace(_BASE, **_SYNAPSES)).eval()
    with pytest.raises(ValueError, match="coincides with the surprise forecast"):
        model(torch.zeros(1, 8, dtype=torch.long), early_readout=1)


@pytest.fixture
def fitting(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import fit_fast_synapses
    return fit_fast_synapses


def test_the_fitted_loss_is_the_loss_with_the_recall_spikes(fitting):
    import math

    from torch.nn import functional as F

    generator = torch.Generator().manual_seed(11)
    data = torch.randint(1, 5, (401,), generator=generator).numpy().astype("uint8")
    surprise = torch.randint(0, 3, (400,), generator=generator).numpy().astype("int16")
    outputs = torch.randint(-40, 40, (400, 256), generator=generator).numpy().astype("int16")
    memory = FastSynapses((1, 2, 3), groups=3, cells=8, levels=2, gate=1)
    scale = math.log(2) / 32
    spikes = torch.randint(-30, 60, (3, 2), generator=generator)
    features = fitting.replay(memory, data, surprise, outputs, scale)
    assert 0 < features.stored_share < 1 and len(features.rows)

    # The decoder adds whole recall spikes to its output counts and is scored on the scaled counts.
    memory.readout[:, 1:] = spikes.to(torch.int32)
    votes, levels, _ = memory(torch.from_numpy(data[:-1].astype("int64"))[None],
                              torch.from_numpy(surprise.astype("int64"))[None])
    counts = torch.from_numpy(outputs.astype("int64")) + memory.excitation(votes, levels)[0]
    labels = torch.from_numpy(data[1:].astype("int64"))
    for start, stop in ((0, 400), (0, 200), (200, 400), (137, 311)):
        expected = F.cross_entropy((counts[start:stop].double() * scale), labels[start:stop], reduction="sum")
        actual = fitting.loss_nats(features, spikes.double() * scale, start, stop)
        assert actual.item() == pytest.approx(expected.item(), rel=1e-5)
    # Fitting can only lower the loss it starts from.
    fitted = fitting.fit(features, 0, 400, iterations=20)
    silent = torch.zeros(3, 2, dtype=torch.float64)
    assert fitting.bpb(features, fitted, 0, 400) < fitting.bpb(features, silent, 0, 400)
