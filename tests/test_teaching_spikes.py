"""The fast output synapses of a stack, which every arriving byte teaches along a stream."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from snnformer.event_lm import _TEACH_LIMIT, _TEACH_UNITS, EventLM, EventLMConfig, TaughtState

_LEVELS = (0, 1)
# Reads that reach 48 cells back, 96 cells after both levels of a small stack, and pools that set the size of a spike.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_cap=128, granule_cap=128, pool_fine=16, pool_rest=4,
              pool_rate=32, output_pool=128)
# Fast synapses from the cells after the upper level. One spike of an output neuron takes 2 ** 24 units of charge.
_TAUGHT = dict(teach_after=(1,), teach_threshold=1 << 24)


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model.eval()


def _bytes(seed=2, length=160, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(seed))


def _pieces(model, tokens, cuts, state=None):
    counts, start = [], 0
    with torch.no_grad():
        for length in cuts:
            fired, state = model.stream(tokens[:, start:start + length], state)
            counts.append(fired)
            start += length
    assert start == tokens.shape[1]
    return torch.cat(counts, 1), state


def _bits(model, counts, tokens):
    """The bits of every byte after the first, as [streams, bytes - 1]."""
    logits = counts[:, :-1].float() * (math.log(2) / (8 * model.cfg.resolution))
    losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none")
    return losses.view(tokens.shape[0], -1) / math.log(2)


@pytest.mark.parametrize("design", [_TAUGHT, dict(_TAUGHT, teach_after=_LEVELS), dict(_TAUGHT, teach_budget=1000),
                                    dict(_TAUGHT, read_delays=4, granule_fan=128, granule_delays=4)])
@pytest.mark.parametrize("cuts", [(80, 80), (48, 48, 64), (47, 1, 49, 63), (7, 13, 100, 40), (1, 2, 3, 154), (130, 30)])
def test_a_taught_stream_cut_anywhere_fires_the_spikes_of_one_pass(design, cuts):
    model, tokens = _stack(**design), _bytes()
    whole, after = _pieces(model, tokens, (160,))
    fired, state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole) and torch.equal(whole, whole.round())
    # The synapses have learned the same from the pieces.
    assert torch.equal(state.taught.fast, after.taught.fast) and torch.equal(state.taught.mean, after.taught.mean)
    assert state.taught.heard == after.taught.heard == 160


def test_a_taught_stream_arrives_a_byte_at_a_time():
    model, tokens = _stack(**_TAUGHT), _bytes(length=60)
    whole, _ = _pieces(model, tokens, (60,))
    fired, _ = _pieces(model, tokens, (1,) * 60)
    assert torch.equal(fired, whole)


def test_the_fast_synapses_are_silent_until_a_byte_has_taught_them():
    taught, plain, tokens = _stack(**_TAUGHT), _stack(), _bytes()
    with torch.no_grad():
        fired, state = taught.stream(tokens)
        silent = plain(tokens)
        # A window has no stream to learn from: there the stack fires as it does without fast synapses.
        assert torch.equal(taught(tokens), silent)
    # The first byte of a stream is predicted before any byte has taught a synapse.
    assert torch.equal(fired[:, 0], silent[:, 0]) and not torch.equal(fired, silent)
    # The fast neurons are a population of their own: their spikes add to the output neurons' counts.
    assert torch.equal(fired - silent, (fired - silent).round()) and (fired - silent).abs().max() <= 4096
    assert isinstance(state.taught, TaughtState) and plain.stream(tokens)[1].taught is None


def test_a_later_byte_teaches_nothing_about_an_earlier_one():
    model, tokens = _stack(**_TAUGHT), _bytes()
    changed = tokens.clone()
    changed[:, 70:] = _bytes(seed=9)[:, 70:]
    first, _ = _pieces(model, tokens, (160,))
    second, _ = _pieces(model, changed, (160,))
    # The count at a byte predicts the next one, so the bytes up to the first changed one fire as before.
    assert torch.equal(first[:, :70], second[:, :70]) and not torch.equal(first[:, 70:], second[:, 70:])


def test_a_state_can_be_continued_more_than_once():
    model, tokens = _stack(**_TAUGHT), _bytes()
    _, state = _pieces(model, tokens[:, :80], (80,))
    held = state.taught.fast.clone()
    first, _ = _pieces(model, tokens[:, 80:], (80,), state)
    other, _ = _pieces(model, _bytes(seed=5)[:, 80:], (80,), state)
    again, _ = _pieces(model, tokens[:, 80:], (30, 50), state)
    # A piece works on its own copy of the synapses: the state that it starts from stays as it was.
    assert torch.equal(state.taught.fast, held)
    assert torch.equal(first, again) and not torch.equal(first, other)
    with pytest.raises(ValueError):
        model.stream(_bytes(streams=3)[:, :10], state)


def test_the_fast_synapses_hold_whole_numbers_of_their_own_stream():
    model, tokens = _stack(**_TAUGHT), _bytes(streams=3)
    _, state = _pieces(model, tokens, (160,))
    taught = state.taught
    assert taught.fast.dtype == taught.mean.dtype == torch.float64
    assert taught.fast.shape == (3, 256, 96) and taught.mean.shape == (3, 96)
    assert torch.equal(taught.fast, taught.fast.round()) and taught.fast.abs().max() <= _TEACH_LIMIT
    assert torch.equal(taught.active, taught.active.round()) and torch.equal(taught.mean, taught.mean.round())
    # The activities of all fibres share the units of a byte, up to the rounding of each.
    assert (taught.active.abs().sum(-1) - _TEACH_UNITS).abs().max() <= 96 / 2
    # Every stream teaches its own synapses: alone it learns what it learns beside the others.
    for row in range(3):
        _, alone = _pieces(model, tokens[row:row + 1], (160,))
        assert torch.equal(alone.taught.fast[0], taught.fast[row])


def test_the_byte_that_came_strengthens_its_own_neuron():
    model, tokens = _stack(**_TAUGHT), _bytes(length=2)
    _, state = _pieces(model, tokens, (1, 1))
    taught, came = state.taught, tokens[:, 1]
    with torch.no_grad():
        _, before = model.stream(tokens[:, :1])
    active = before.taught.active
    rows = torch.arange(2)
    # The neuron of the byte that came is taught more spikes than it fired, and the others fired without it.
    gained = taught.fast[rows, came] / active
    assert (gained[active != 0] > 0).all()
    others = taught.fast.clone()
    others[rows, came] = 0
    lost = others / active[:, None, :]
    assert (lost[(active != 0)[:, None, :].expand_as(lost)] <= 0).all()
    # A change is the fibre's activity times a whole number of spikes, and the race fires close to its budget.
    spikes = taught.fast[:, :, (active != 0).all(0)] / active[:, None, (active != 0).all(0)]
    assert torch.equal(spikes, spikes.round()) and (spikes.sum(1) - spikes.sum(1)[:, :1] == 0).all()
    assert spikes.sum(1).abs().max() <= 256


def test_the_fast_synapses_learn_a_stream_that_repeats():
    pattern = _bytes(seed=11, length=24, streams=2)
    tokens = pattern.repeat(1, 20)
    taught, plain = _stack(**_TAUGHT), _stack()
    learned, _ = _pieces(taught, tokens, (480,))
    with torch.no_grad():
        frozen, _ = plain.stream(tokens)
    late = slice(360, 479)
    with_teaching, without = _bits(taught, learned, tokens)[:, late].mean(), _bits(plain, frozen, tokens)[:, late].mean()
    # An untrained stack predicts none of it. With teaching it has learned most of the repeated bytes.
    assert without > 7 and with_teaching < without - 3
    # A higher threshold lets the same synapses say less.
    quiet = _stack(**dict(_TAUGHT, teach_threshold=1 << 34))
    hushed, _ = _pieces(quiet, tokens, (480,))
    assert with_teaching < _bits(quiet, hushed, tokens)[:, late].mean() <= without + 0.01


def test_a_trained_stack_is_given_its_fast_synapses_afterwards():
    model, built, tokens = _stack(), _stack(**_TAUGHT), _bytes()
    plain, _ = _pieces(model, tokens, (160,))
    model.teach((1,), 1 << 24)
    assert model.cfg == built.cfg and set(model.state_dict()) == set(built.state_dict())
    taught, _ = _pieces(model, tokens, (160,))
    assert torch.equal(taught, _pieces(built, tokens, (160,))[0]) and not torch.equal(taught, plain)
    model.teach(None)
    assert model.cfg.teach_after is None and torch.equal(_pieces(model, tokens, (160,))[0], plain)


@pytest.mark.parametrize("setting", [dict(teach_after=(2,)), dict(teach_after=()), dict(teach_after=(1, 1)),
                                     dict(teach_threshold=0), dict(teach_threshold=1.5), dict(teach_threshold=True),
                                     dict(teach_budget=0), dict(teach_budget=4097),
                                     dict(granule_after=(0,)), dict(stack=False, read_after=None, granule_after=None,
                                                                    read_cap=None, granule_cap=None, pool_fine=0,
                                                                    pool_rest=0, pool_rate=0, output_pool=0)])
def test_fast_output_synapses_need_the_cells_of_a_stack(setting):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **dict(_TAUGHT, **setting))))
    model = _stack()
    with pytest.raises(ValueError):
        model.teach((5,), 1 << 24)
    assert model.cfg.teach_after is None and model.teach_after is None


def test_the_scorer_teaches_every_stream_by_itself(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_stream

    data = torch.from_numpy(np.random.default_rng(1).integers(0, 256, 1000, dtype=np.uint8))
    model = _stack()
    streams = eval_stream.stream_rows(data, 4, 30)
    plain, count = eval_stream.score(model, streams, 50, torch.device("cpu"))
    model.teach((1,), 1 << 24)
    taught, _ = eval_stream.score(model, streams, 50, torch.device("cpu"))
    assert count == 999 and taught.shape == plain.shape and not np.array_equal(taught, plain)
    # The pieces do not change what is learned or scored.
    other, _ = eval_stream.score(model, streams, 273, torch.device("cpu"))
    assert taught.sum() == pytest.approx(other.sum(), rel=1e-12)
