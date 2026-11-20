"""Fast output synapses that training runs in every window as a stream runs them, so that the slow weights learn to
make the learning that happens while reading useful, and that a stream then runs in whole numbers with the gains
that training found."""

from pathlib import Path

import numpy as np
import pytest
import torch

from snnformer.event_lm import _TEACH_LIMIT, EventLM, EventLMConfig, _earlier_bytes
from test_teaching_spikes import _STACK, _bytes, _pieces

_PLASTIC = dict(plastic_after=(1,))


def _stack(seed=0, gains=None, dtype=torch.float32, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
        if gains is not None:
            model.plastic_gains.copy_(torch.tensor(gains))
    return model.to(dtype).eval()


@pytest.mark.parametrize("design, gains", [(_PLASTIC, [1.0]), (dict(plastic_after=(0, 1)), [0.5, -2.0]),
                                           (dict(_PLASTIC, plastic_shuffle=True), [1.0]),
                                           (dict(_PLASTIC, plastic_threshold=1 << 20), [0.25]),
                                           (dict(_PLASTIC, teach_budget=1000), [1.0])])
def test_a_window_fires_the_spikes_of_the_stream_that_it_starts(design, gains):
    # In 64-bit floats and with gains whose thresholds are whole, the parallel form is the stream's rule exactly.
    model, tokens = _stack(gains=gains, dtype=torch.float64, **design), _bytes()
    with torch.no_grad():
        window, plain = model(tokens), _stack(dtype=torch.float64)(tokens)
    streamed, _ = _pieces(model, tokens, (160,))
    assert torch.equal(window, streamed) and torch.equal(window, window.round())
    assert (window - plain).abs().sum() > 0


def test_a_window_in_32_bit_floats_and_with_a_decay_fires_them_within_rounding():
    for design, gains, dtype in ((dict(plastic_after=(0, 1)), [0.5, -2.0], torch.float32),
                                 (dict(_PLASTIC, plastic_decay_shift=6), [1.0], torch.float64)):
        model, tokens = _stack(gains=gains, dtype=dtype, **design), _bytes()
        with torch.no_grad():
            window = model(tokens)
        streamed, _ = _pieces(model, tokens, (160,))
        differs = (window - streamed).abs()
        assert differs.max() <= 1 and (differs > 0).float().mean() < 0.02


def test_a_window_longer_than_the_exact_mean_keeps_to_the_stream_within_rounding():
    model, tokens = _stack(gains=[1.0], dtype=torch.float64, **_PLASTIC), _bytes(length=1100, streams=1)
    with torch.no_grad():
        window = model(tokens)
    streamed, _ = _pieces(model, tokens, (1100,))
    assert torch.equal(window[:, :1025], streamed[:, :1025])
    differs = (window - streamed).abs()
    assert differs.max() <= 2 and (differs > 0).float().mean() < 0.02


@pytest.mark.parametrize("design", [_PLASTIC, dict(plastic_after=(0, 1), plastic_decay_shift=5),
                                    dict(_PLASTIC, plastic_shuffle=True)])
def test_a_later_byte_changes_no_spike_before_it(design):
    model, tokens = _stack(gains=[1.0] * len(design["plastic_after"]), **design), _bytes()
    changed = tokens.clone()
    changed[:, 71:] = _bytes(seed=9)[:, 71:]
    with torch.no_grad():
        first, second = model(tokens), model(changed)
    # The count at byte 70 predicts byte 71, so it and every count before it stand as they were.
    assert torch.equal(first[:, :71], second[:, :71]) and not torch.equal(first[:, 71:], second[:, 71:])


@pytest.mark.parametrize("design", [_PLASTIC, dict(plastic_after=(0, 1), plastic_decay_shift=5),
                                    dict(_PLASTIC, plastic_shuffle=True)])
def test_a_stream_cut_into_pieces_of_1_7_or_512_bytes_fires_and_learns_the_same(design):
    model, tokens = _stack(gains=[0.5] * len(design["plastic_after"]), **design), _bytes(length=520)
    whole, after = _pieces(model, tokens, (520,))
    for cuts in ((1,) * 520, (7,) * 74 + (2,), (512, 8)):
        fired, state = _pieces(model, tokens, cuts)
        assert torch.equal(fired, whole)
        for name in ("fast", "mean", "total", "errors"):
            mine, theirs = getattr(state.taught, name), getattr(after.taught, name)
            assert (mine is None and theirs is None) or torch.equal(mine, theirs)


def test_at_gains_of_zero_the_stack_is_the_one_without_fast_synapses():
    plain, trained, tokens = _stack(), _stack(plastic_after=(0, 1)), _bytes()
    assert trained.plastic_gains.tolist() == [0.0, 0.0]
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    assert count(trained) == count(plain) + 2
    with torch.no_grad():
        assert torch.equal(trained(tokens), plain(tokens))
    assert torch.equal(_pieces(trained, tokens, (160,))[0], _pieces(plain, tokens, (160,))[0])


def test_a_frozen_stream_keeps_its_synapses_and_fires_with_them():
    model, tokens = _stack(gains=[1.0], **_PLASTIC), _bytes()
    learning, _ = _pieces(model, tokens, (160,))
    _, early = _pieces(model, tokens[:, :60], (60,))
    model.freeze_teaching_after(60)
    frozen, state = _pieces(model, tokens, (100, 60))
    # The first 60 bytes teach as before. The bytes after them change no synapse, and the synapses keep firing.
    assert torch.equal(state.taught.fast, early.taught.fast)
    assert torch.equal(frozen[:, :60], learning[:, :60]) and not torch.equal(frozen, learning)
    with torch.no_grad():
        model.silence_plasticity()
        silent = model(tokens)
    assert (frozen[:, 60:] - silent[:, 60:]).abs().sum() > 0
    with pytest.raises(ValueError):
        model.freeze_teaching_after(-1)


def test_the_gains_and_the_cells_learn_through_the_fast_spikes_and_the_errors_carry_no_gradient():
    tokens, seen = _bytes(), {}
    for gain in (0.0, 1.0):
        model = _stack(gains=[gain], **_PLASTIC).train()
        pulses = []
        counts = model(tokens, memory_pulses=pulses)
        # A loss that is a sum of the counts: its gradient is that of the stack's own counts plus the fast spikes'.
        counts.sum().backward()
        seen[gain] = dict(gain=model.plastic_gains.grad.clone(), cells=model.granule_layers["1"].receive.grad.clone(),
                          output=model.output.weight.grad.clone(), pulses=pulses[-1].item(), counts=counts.detach())
    silent, loud = seen[0.0], seen[1.0]
    # At a gain of zero the fast spikes are silent, and the gain already knows which way to move.
    assert silent["pulses"] == 0 and silent["gain"].abs().sum() > 0
    # With a gain the cells learn through the fast spikes as well.
    assert not torch.equal(silent["cells"], loud["cells"])
    # The output synapses set the counts that decide what a byte teaches, and that teaching passes no gradient.
    assert torch.equal(silent["output"], loud["output"])
    # The fast spikes count among the pulses that a budget prices.
    assert loud["pulses"] == pytest.approx((loud["counts"] - silent["counts"]).abs().sum().item())


def test_the_gains_are_not_weight_decayed(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    from common import make_optimizer

    model = _stack(plastic_after=(0, 1))
    for rotated in (None, 50):
        optimizer = make_optimizer(model, 1e-3, 0.05, rotated=rotated, stacks=True)
        group = next(group for group in optimizer.param_groups
                     if any(parameter is model.plastic_gains for parameter in group["params"]))
        assert group["weight_decay"] == 0.0 and model.plastic_gains.ndim == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_the_stream_rule_refuses_mps():
    model = _stack(gains=[1.0], **_PLASTIC).to("mps")
    with pytest.raises(ValueError, match="MPS"):
        model.stream(_bytes().to("mps"))


def test_the_shuffled_control_teaches_with_errors_of_earlier_bytes_only():
    earlier = _earlier_bytes(0, 3000)
    positions = torch.arange(3000)
    assert earlier[0] == -1 and (earlier[1:] < positions[1:]).all()
    assert (earlier[1:] >= (positions[1:] - 1024).clamp_min(0)).all()
    # A stream continued from any byte draws what a window from the start draws.
    assert torch.equal(_earlier_bytes(700, 50), earlier[700:750])
    assert len(set(earlier[1100:2100].tolist())) > 500


def test_trained_synapses_stay_whole_and_bounded():
    model, tokens = _stack(gains=[1.0], **dict(_PLASTIC, plastic_decay_shift=8)), _bytes(streams=3)
    _, state = _pieces(model, tokens, (160,))
    taught = state.taught
    assert torch.equal(taught.fast, taught.fast.round()) and taught.fast.abs().max() <= _TEACH_LIMIT
    assert torch.equal(taught.mean, taught.mean.round()) and torch.equal(taught.total, taught.total.round())


@pytest.mark.parametrize("setting", [dict(plastic_after=(2,)), dict(plastic_after=()), dict(plastic_after=(1, 1)),
                                     dict(plastic_after=(1,), teach_after=(1,), teach_threshold=1 << 24),
                                     dict(plastic_after=(1,), plastic_rule="delta"),
                                     dict(plastic_after=(1,), plastic_decay_shift=31),
                                     dict(plastic_after=(1,), plastic_threshold=0),
                                     dict(plastic_decay_shift=4), dict(plastic_shuffle=True),
                                     dict(teach_freeze_after=-3), dict(granule_after=(0,), plastic_after=(1,))])
def test_trained_fast_synapses_need_the_cells_of_a_stack(setting):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **setting)))


def test_a_trained_stack_takes_no_fast_synapses_after_training_and_can_be_silenced():
    model, plain, tokens = _stack(gains=[1.0], **_PLASTIC), _stack(), _bytes()
    with pytest.raises(ValueError):
        model.teach((1,), 1 << 24)
    model.silence_plasticity()
    with torch.no_grad():
        assert torch.equal(model(tokens), plain(tokens))
    assert torch.equal(_pieces(model, tokens, (160,))[0], _pieces(plain, tokens, (160,))[0])


def test_the_scorer_runs_the_trained_rule_unless_it_is_switched_off(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_stream

    data = torch.from_numpy(np.random.default_rng(1).integers(0, 256, 1000, dtype=np.uint8))
    streams = eval_stream.stream_rows(data, 4, 30)
    model = _stack(gains=[1.0], **_PLASTIC)
    trained, count = eval_stream.score(model, streams, 50, torch.device("cpu"))
    other, _ = eval_stream.score(model, streams, 273, torch.device("cpu"))
    model.silence_plasticity()
    silent, _ = eval_stream.score(model, streams, 50, torch.device("cpu"))
    plain, _ = eval_stream.score(_stack(), streams, 50, torch.device("cpu"))
    assert count == 999 and not np.array_equal(trained, silent) and np.array_equal(silent, plain)
    # The pieces do not change what is learned or scored.
    assert trained.sum() == pytest.approx(other.sum(), rel=1e-12)
