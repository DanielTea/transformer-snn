"""A stack that sleeps: after every window of a stream it replays the window and rewrites its reads' tapes."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import test_lifted_ceiling
from snnformer.event_lm import _SLEEP_QUARTERS, EventLMConfig, ReadTape
from test_teaching_spikes import _bytes, _pieces

_WINDOW = 16


def _stack(lifted=True, seed=0, **design):
    return test_lifted_ceiling._stack(seed, **dict(design, pool_lifts_ceiling=lifted)).double()


def _reads(model):
    return list(dict.fromkeys(model.state_reads.values()))


def _gates(model, rest):
    """Hold every gate cell at a rest of ``rest`` charge: far above zero it keeps the stored cells, far below it takes
    the replay's."""
    with torch.no_grad():
        for read in _reads(model):
            read.sleep_rest.fill_(rest / read.rate)


def test_sleep_is_off_unless_asked_for_and_adds_only_the_replay_parts():
    assert EventLMConfig.stack_sleep == 0
    plain, sleeping = _stack(), _stack(stack_sleep=_WINDOW)
    added = set(sleeping.state_dict()) - set(plain.state_dict())
    assert added and all(name.endswith(("ahead", "sleep_gate", "sleep_rest")) for name in added)
    assert set(plain.state_dict()) <= set(sleeping.state_dict())
    for read in _reads(sleeping):
        assert read.ahead.shape == (read.heads, _WINDOW) and not read.sleep_gate.any() and not read.sleep_rest.any()


@pytest.mark.parametrize("design", [dict(), dict(lifted=False), dict(stack_loop=(0, 1, 2))])
def test_a_sleeping_stack_fires_what_its_window_fires_however_it_is_cut(design):
    model, tokens = _stack(**design, stack_sleep=_WINDOW).eval(), _bytes(length=100)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (100,))
    assert torch.equal(window, whole) and torch.equal(whole, whole.round())
    for cuts in ((1,) * 100, (7,) * 14 + (2,), (50, 50), (15, 17, 68), (16, 16, 68)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    model._check_accumulator()


@pytest.mark.parametrize("design", [dict(), dict(stack_loop=(0, 1, 2))])
def test_sleep_changes_nothing_before_the_first_window_ends(design):
    tokens = _bytes(length=60)
    plain, sleeping = _stack(**design).eval(), _stack(**design, stack_sleep=_WINDOW).eval()
    sleeping.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        before, after = plain(tokens), sleeping(tokens)
    assert torch.equal(before[:, :_WINDOW], after[:, :_WINDOW])
    assert not torch.equal(before[:, _WINDOW:], after[:, _WINDOW:])


@pytest.mark.parametrize("design", [dict(), dict(stack_loop=(0, 1, 2))])
def test_gates_that_keep_every_cell_leave_the_stack_as_it_was(design):
    tokens = _bytes(length=70)
    plain, sleeping = _stack(**design).eval(), _stack(**design, stack_sleep=_WINDOW).eval()
    sleeping.load_state_dict(plain.state_dict(), strict=False)
    _gates(sleeping, 40.0)
    with torch.no_grad():
        assert torch.equal(sleeping(tokens), plain(tokens))


def test_open_gates_write_the_replay_into_the_tape():
    tokens = _bytes(length=_WINDOW)
    model = _stack(stack_sleep=_WINDOW).eval()
    _gates(model, -40.0)
    with torch.no_grad():
        _, awake = _pieces(model, tokens[:, :-1], (_WINDOW - 1,))
        _, slept = model.stream(tokens[:, -1:], awake)
    for key, read in model.state_reads.items():
        tape, before = slept.reads[key], awake.reads[key]
        # The cells before the window's last byte were rewritten by the replay, in whole pulses within the cap.
        assert not torch.equal(tape.tag[:, :, :-1], before.tag) or not torch.equal(tape.payload[:, :, :-1], before.payload)
        for cells in (tape.tag, tape.payload):
            assert torch.equal(cells, cells.round()) and cells.abs().max() <= read.fire.capped
    # The window that waits for its replay was taken by it.
    assert slept.awake is None and slept.awake_tokens is None and awake.awake.shape[1] == _WINDOW - 1


def test_the_replay_hears_the_later_bytes_of_its_window():
    tokens = _bytes(length=_WINDOW)
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 256
    model = _stack(stack_sleep=_WINDOW).eval()
    _gates(model, -40.0)
    with torch.no_grad():
        first = model.stream(tokens)[1].reads["1"]
        second = model.stream(changed)[1].reads["1"]
    # The window's first cells, on the second level's tape, depend on the window's last byte.
    assert not torch.equal(first.payload[:, :, :4], second.payload[:, :, :4])


def test_a_gate_keeps_whole_quarters_of_the_stored_cell():
    model = _stack(stack_sleep=_WINDOW)
    read = _reads(model)[0]
    old = torch.full((1, read.heads, 3, read.sleep_gate.shape[-1] // 2), 8.0, dtype=torch.float64)
    made = torch.zeros_like(old)
    for rest, kept in ((0.0, 2), (40.0, _SLEEP_QUARTERS), (-40.0, 0)):
        tape = ReadTape(tag=old.clone(), payload=old.clone())
        _gates(model, rest)
        read._blend(tape, made, made)
        assert torch.equal(tape.tag, torch.full_like(old, 8.0 * kept / _SLEEP_QUARTERS))
        assert torch.equal(tape.payload, tape.tag)


def test_later_windows_learn_through_the_replays():
    model, tokens = _stack(stack_sleep=_WINDOW).train(), _bytes(length=3 * _WINDOW)
    counts, state = model.stream(tokens)
    counts.float().square().mean().backward()
    for read in _reads(model):
        for weights in (read.sleep_gate, read.sleep_rest):
            assert weights.grad is not None and torch.isfinite(weights.grad).all() and weights.grad.abs().sum() > 0
    # The lower read's race in the replay reaches the cells above it. The top read only writes its cells: what its
    # race would pass on reaches nothing.
    lower, top = model.state_reads["0"], model.state_reads["1"]
    assert lower.ahead.grad is not None and lower.ahead.grad.abs().sum() > 0
    assert top.ahead.grad is None
    # The state handed on keeps no gradient.
    for tape in state.reads.values():
        assert not tape.attached and not tape.tag.requires_grad and not tape.payload.requires_grad
    assert all(not fibres.attached for fibres in state.cells.values())


def test_a_piece_learns_nothing_through_a_replay_whose_window_it_does_not_read():
    model, tokens = _stack(stack_sleep=_WINDOW).train(), _bytes(length=_WINDOW)
    counts, _ = model.stream(tokens)
    counts.float().square().mean().backward()
    # The only replay came after the piece's last byte, so its gate cells and offsets learned nothing.
    for read in _reads(model):
        assert read.sleep_gate.grad is None or not read.sleep_gate.grad.any()
        assert read.ahead.grad is None or not read.ahead.grad.any()


@pytest.mark.parametrize("setting", [
    dict(stack_sleep=-1), dict(stack_sleep=True), dict(stack_sleep=1), dict(stack_sleep=48), dict(stack_sleep=2.0),
    dict(stack_sleep=_WINDOW, read_mode="soft"), dict(stack_sleep=_WINDOW, read_kept=4),
    dict(stack_sleep=_WINDOW, stack_skips="level"),
])
def test_a_stack_refuses_a_sleep_it_cannot_have(setting):
    with pytest.raises(ValueError, match="sleeps"):
        _stack(**setting)


def test_a_sleeping_stack_takes_its_bytes_only_as_a_stream():
    model, tokens = _stack(stack_sleep=_WINDOW), _bytes(length=20)
    with pytest.raises(ValueError, match="sleeping stack"):
        model(tokens, return_prediction=True)


def test_the_trainer_hands_the_sleep_to_the_model(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    flags = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 2 --spike-resolution 16
    --router-charge-bits 8 --read-after 0,1 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --granule-after 0,1 --granule-clusters 1 --granule-fan 32
    --granule-cells 16 --granule-slope --pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128
    --pool-lifts-ceiling --read-delays 4 --granule-delays 4 --stack-sleep 4 --pulse-budget 100000 --ctx 16 --bs 2
    --steps 4 --stream 2 --warmup 1 --lr 0.003 --seed 3 --eval-every 4 --eval-bs 2 --eval-batches 1
    --final-eval-batches 1 --save-every 2 --validation-only""".split()
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *flags, "--checkpoint", str(tmp_path / "model.pt"),
                                      "--results-dir", str(tmp_path / "results")])
    train_event_lm.main()
    model, _ = train_event_lm.load_checkpoint(tmp_path / "model.pt", torch.device("cpu"))
    assert model.cfg.stack_sleep == 4
    read = model.state_reads["0"]
    assert read.sleep == 4 and read.ahead.shape[-1] == 4
    # Four updates on streams moved the gates: later windows of every piece read what the replays wrote.
    assert read.sleep_rest.any()


def test_a_sleeping_stack_checks_its_weights_once_for_all_stretches(monkeypatch):
    model, tokens = _stack(stack_sleep=_WINDOW), _bytes(length=70)
    model._check_weights = True
    calls = []
    original = model._check_accumulator
    monkeypatch.setattr(model, "_check_accumulator", lambda: calls.append(1) or original())
    with torch.no_grad():
        model(tokens)
    # Five stretches, one check; and the setting is as it was.
    assert len(calls) == 1 and model._check_weights is True
