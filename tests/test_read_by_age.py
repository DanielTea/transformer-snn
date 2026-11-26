"""A stream's budgeted race read computed by age (_StateRead._address_band) gives the spikes of the read by cell
exactly, and the same gradient up to rounding."""

import pytest
import torch

import snnformer.event_lm as event_lm
import test_knowledge_gap_cells
from test_knowledge_gap_cells import _bytes

_DESIGNS = [dict(), dict(read_knowledge=8), dict(read_reach=96), dict(read_reach=96, read_knowledge=8),
            dict(read_reach=96, read_knowledge=8, read_forget=True),
            dict(read_reach=96, read_knowledge=8, read_forget=True, stack_loop=(0, 1, 2), granule_power=8)]


def _stack(seed=0, **design):
    return test_knowledge_gap_cells._stack(seed, **design)


def _stream(model, tokens, cuts, by_age):
    """The stream's counts and its last state, with the reads by age or by cell."""
    saved = event_lm._READ_BY_AGE
    event_lm._READ_BY_AGE = by_age
    try:
        counts, state, start = [], None, 0
        for length in cuts:
            fired, state = model.stream(tokens[:, start:start + length], state)
            counts.append(fired)
            start += length
        return torch.cat(counts, 1), state
    finally:
        event_lm._READ_BY_AGE = saved


@pytest.mark.parametrize("design", _DESIGNS)
@pytest.mark.parametrize("cuts", [(160,), (40, 40, 40, 40), (7, 96, 57), (1,) * 12 + (148,)])
@pytest.mark.parametrize("block", [256, 16, 5])
def test_reads_by_age_fire_what_reads_by_cell_fire(design, cuts, block, monkeypatch):
    # Blocks of 256 bytes take a window whole here; smaller ones cut it, at the reach's edge and within it.
    monkeypatch.setattr(event_lm, "_AGE_BLOCK", block)
    model, tokens = _stack(**design).eval(), _bytes(length=160)
    with torch.no_grad():
        by_cell, cell_state = _stream(model, tokens, cuts, by_age=False)
        by_age, age_state = _stream(model, tokens, cuts, by_age=True)
    assert torch.equal(by_age, by_cell)
    for name, tape in cell_state.reads.items():
        other = age_state.reads[name]
        assert torch.equal(other.payload, tape.payload) and torch.equal(other.tag, tape.tag)


def test_reads_by_age_are_used_by_streams_and_fire_what_a_window_fires():
    model, tokens = _stack(read_reach=96, read_knowledge=8).eval(), _bytes(length=120)
    calls = []
    original = event_lm._StateRead._address_band

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    event_lm._StateRead._address_band = counted
    try:
        with torch.no_grad():
            window = model(tokens)
            assert not calls
            stream, _ = _stream(model, tokens, (50, 70), by_age=True)
    finally:
        event_lm._StateRead._address_band = original
    assert calls and torch.equal(stream, window)


@pytest.mark.parametrize("design", [dict(read_reach=96, read_knowledge=8, read_forget=True),
                                    dict(read_reach=96, read_knowledge=8, stack_loop=(0, 1, 2))])
@pytest.mark.parametrize("block", [256, 16])
def test_reads_by_age_learn_with_the_gradient_of_reads_by_cell(design, block, monkeypatch):
    monkeypatch.setattr(event_lm, "_AGE_BLOCK", block)
    grads = []
    for by_age in (False, True):
        model, tokens = _stack(**design).double().train(), _bytes(length=120)
        state = None
        saved = event_lm._READ_BY_AGE
        event_lm._READ_BY_AGE = by_age
        try:
            with torch.no_grad():
                _, state = model.stream(tokens[:, :64], state)
            counts, _ = model.stream(tokens[:, 64:], state)
        finally:
            event_lm._READ_BY_AGE = saved
        weights = torch.linspace(-1, 1, counts.numel(), dtype=counts.dtype).reshape(counts.shape)
        (counts * weights).sum().backward()
        grads.append({name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()})
    by_cell, by_age = grads
    read_grads = 0
    for name, grad in by_cell.items():
        other = by_age[name]
        assert (grad is None) == (other is None), name
        if grad is not None:
            assert torch.allclose(other, grad, rtol=1e-9, atol=1e-12), name
            read_grads += "state_reads" in name and bool(grad.abs().sum() > 0)
    assert read_grads > 0


def test_ages_in_reach_mark_the_cells_before_a_stream_and_give_each_cell_its_phase():
    back, valid, phase = event_lm._ages_in_reach(3, 4, 5, 10, 2, torch.device("cpu"), torch.float64)
    assert back.tolist()[0] == [3, 2, 1, 0, -1] and valid.tolist()[0] == [True, True, True, True, False]
    spread = event_lm._spread_phases(10 - 3, 3 + 4, torch.device("cpu"), torch.float64)
    for head in range(2):
        for byte in range(4):
            for age in range(5):
                cell = 3 + byte - age
                if cell >= 0:
                    assert phase[head, byte, age] == (spread[cell] + head / 2) % 1


@pytest.mark.parametrize("knowledge", [True, False])
def test_taught_streams_hear_the_same_knowledge_cells_by_age_and_by_cell(knowledge, monkeypatch):
    monkeypatch.setattr(event_lm, "_AGE_BLOCK", 16)
    model, tokens = _stack(read_reach=96, read_knowledge=8 if knowledge else 0).eval(), _bytes(length=160)
    model.teach((0, 1), 1 << 22, knowledge=knowledge)
    with torch.no_grad():
        by_cell, _ = _stream(model, tokens, (40, 40, 80), by_age=False)
        by_age, _ = _stream(model, tokens, (40, 40, 80), by_age=True)
    assert torch.equal(by_age, by_cell)
