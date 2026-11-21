"""A loop of a stack's levels that every byte passes several times with the same reads and cell layers."""

from dataclasses import asdict

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig
from test_teaching_spikes import _STACK, _bytes, _pieces

_LOOP = dict(_STACK, stack_loop=(0, 1, 3))


def _stack(seed=0, dtype=torch.float32, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses, once for each level of the stack.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in dict.fromkeys(model.state_reads.values()):
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in dict.fromkeys(model.granule_layers.values()):
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model.to(dtype).eval()


def test_one_pass_is_the_plain_stack():
    plain, once, tokens = _stack(), _stack(stack_loop=(0, 1, 1)), _bytes()
    assert [name for name, _ in once.named_parameters()] == [name for name, _ in plain.named_parameters()]
    with torch.no_grad():
        assert torch.equal(once(tokens), plain(tokens))


def test_the_passes_of_a_level_share_its_read_and_its_cells():
    plain, loop = _stack(), _stack(**_LOOP)
    assert loop.visits == [0, 1, 0, 1, 0, 1]
    for later, first in ((2, 0), (4, 0), (3, 1), (5, 1)):
        assert loop.state_reads[str(later)] is loop.state_reads[str(first)]
        assert loop.granule_layers[str(later)] is loop.granule_layers[str(first)]
    count = lambda model: sum(parameter.numel() for parameter in model.parameters())  # noqa: E731
    assert count(loop) == count(plain)
    # A loop inside a longer stack: the levels before and after it run once.
    inner = EventLM(EventLMConfig(**dict(_STACK, depth=4, read_after=(0, 1, 2, 3), granule_after=(0, 1, 2, 3),
                                         stack_loop=(1, 2, 2))))
    assert inner.visits == [0, 1, 2, 1, 2, 3] and inner.state_reads["3"] is inner.state_reads["1"]


def test_every_pass_trains_the_level_it_passes():
    # The same weights unrolled into a stack whose passes are levels of their own: it fires the same, and the
    # gradient of a shared weight is the sum of what its passes take.
    loop = _stack(dtype=torch.float64, stack_loop=(0, 1, 2)).train()
    unrolled = EventLM(EventLMConfig(**dict(_STACK, depth=4, read_after=(0, 1, 2, 3), granule_after=(0, 1, 2, 3))))
    unrolled = unrolled.to(torch.float64).train()
    unrolled.load_state_dict(loop.state_dict())
    tokens = _bytes()
    first, second = loop(tokens), unrolled(tokens)
    assert torch.equal(first, second)
    first.square().mean().backward()
    second.square().mean().backward()
    for name, parameter in loop.named_parameters():
        if name.startswith(("state_reads.", "granule_layers.")):
            index = int(name.split(".")[1])
            twin = name.replace(f".{index}.", f".{index + 2}.", 1)
            taken = dict(unrolled.named_parameters())
            assert torch.allclose(parameter.grad, taken[name].grad + taken[twin].grad)
        elif parameter.grad is not None:
            assert torch.allclose(parameter.grad, dict(unrolled.named_parameters())[name].grad)


def test_a_looped_stream_fires_what_its_window_fires_however_it_is_cut():
    model, tokens = _stack(dtype=torch.float64, **_LOOP), _bytes(length=200)
    with torch.no_grad():
        window = model(tokens)
    whole, after = _pieces(model, tokens, (200,))
    assert torch.equal(window, whole)
    # Every pass of a level keeps its own state in the stream.
    assert sorted(after.reads) == [str(index) for index in range(6)]
    for cuts in ((1,) * 200, (7,) * 28 + (4,), (128, 72)):
        fired, _ = _pieces(model, tokens, cuts)
        assert torch.equal(fired, whole)


def test_a_saved_looped_stack_is_rebuilt_from_its_configuration():
    model, tokens = _stack(**_LOOP), _bytes()
    assert model.cfg.stack_loop == (0, 1, 3) and model.cfg.depth == 2
    rebuilt = EventLM(EventLMConfig(**asdict(model.cfg)))
    rebuilt.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.equal(rebuilt.eval()(tokens), model(tokens))


def test_fast_synapses_hear_the_passes_of_a_loop():
    model, tokens = _stack(dtype=torch.float64, **_LOOP), _bytes()
    # Level 5 is the last pass of the stack's level 1: fast synapses may hear any pass.
    model.teach((5,), 1 << 24)
    taught, _ = _pieces(model, tokens, (160,))
    assert model.teach_after == (5,)
    plain, _ = _pieces(_stack(dtype=torch.float64, **_LOOP), tokens, (160,))
    assert not torch.equal(taught, plain)
    with pytest.raises(ValueError):
        model.teach((6,), 1 << 24)


@pytest.mark.parametrize("loop", [(0, 2, 2), (1, 0, 2), (0, 1, 0), (-1, 1, 2), (0, 1), (0.0, 1, 2)])
def test_a_loop_runs_levels_of_the_stack_that_it_has(loop):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, stack_loop=loop)))


def test_a_loop_needs_a_stack():
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(stack_loop=(0, 1, 2)))


def test_the_trainer_trains_a_looped_stack(monkeypatch, tmp_path):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / "loop", "--stack-loop", "0,1,2")
    assert record["args"]["stack_loop"] == "0,1,2" and record["cfg"]["stack_loop"] == [0, 1, 2]
    saved = torch.load(tmp_path / "loop" / "model.pt", weights_only=True)
    rebuilt = EventLM(EventLMConfig(**saved["cfg"]))
    rebuilt.load_state_dict(saved["state_dict"])
    assert rebuilt.visits == [0, 1, 0, 1]


def test_the_scorer_sets_a_shared_read_once_for_all_its_passes(monkeypatch):
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    sys.modules.pop("eval_stream", None)
    import eval_stream

    model = _stack(**_LOOP)
    window = model.state_reads["0"].window
    eval_stream.lengthen_reach(model, 4 * window, (0, 2, 4))
    assert model.state_reads["2"].reach == 4 * window and model.state_reads["1"].reach == window
    eval_stream.forget_by_power(model, -1.0)
    assert all(read.forget is not None for read in model.state_reads.values())
    # Naming one pass of a looped level would set its other passes too, so it is refused.
    with pytest.raises(ValueError, match="passes"):
        eval_stream.lengthen_reach(_stack(**_LOOP), 4 * window, (0, 2))


def test_a_loop_keeps_the_full_stacks_delayed_fibres_and_forgetting_exact_in_a_stream():
    # The released stack's options: cells and tags that hear every channel again one to three bytes later, and
    # reads that learn a forgetting slope and reach further in a stream than their window.
    model = _stack(dtype=torch.float64, **dict(_LOOP, granule_fan=128, granule_delays=4, read_delays=4,
                                               read_forget=True, read_reach=96))
    with torch.no_grad():
        for read in dict.fromkeys(model.state_reads.values()):
            read.forget.copy_(torch.linspace(-1.0, 0.5, read.heads))
    tokens = _bytes(length=200)
    with torch.no_grad():
        window = model(tokens)
    whole, _ = _pieces(model, tokens, (200,))
    assert torch.equal(window, whole)
    for cuts in ((1,) * 200, (7,) * 28 + (4,), (128, 72)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
