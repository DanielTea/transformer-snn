"""A racing loop: the output neurons hear the end of every pass of a loop and sum what they hear as evidence."""

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig
from test_stack_loop import _stack
from test_teaching_spikes import _STACK, _bytes, _pieces


def test_without_exits_the_loop_is_as_it_was():
    plain, racing, tokens = _stack(stack_loop=(0, 1, 3)), _stack(stack_loop=(0, 1, 3), stack_loop_exits=True), _bytes()
    assert [name for name, _ in racing.named_parameters()] == [name for name, _ in plain.named_parameters()]
    assert plain.exits == () and racing.exits == (1, 3)
    with torch.no_grad():
        top = plain(tokens)
        summed = racing(tokens)
    assert plain.exit_counts == [] and len(racing.exit_counts) == 2
    # The top's counts are the plain loop's; the racing loop adds the evidence of its first two passes to them.
    torch.testing.assert_close(summed, top + racing.exit_counts[-1], rtol=0, atol=0)
    first, second = racing.exit_counts
    assert first.shape == top.shape and not torch.equal(first, second)


def test_the_evidence_of_a_pass_is_what_the_output_neurons_fire_for_its_stream():
    racing, tokens = _stack(stack_loop=(0, 1, 2), stack_loop_exits=True), _bytes()
    heard = {}
    hook = racing.granule_layers["1"].register_forward_hook(lambda module, inputs, output: heard.setdefault("end", output))
    with torch.no_grad():
        racing(tokens)
    hook.remove()
    with torch.no_grad():
        torch.testing.assert_close(racing.exit_counts[0], racing._emitted(heard["end"]), rtol=0, atol=0)


def test_a_racing_loop_streams_the_same_however_the_stream_is_cut():
    racing, tokens = _stack(stack_loop=(0, 1, 3), stack_loop_exits=True), _bytes()
    with torch.no_grad():
        whole, _ = racing.stream(tokens)
    cut, _ = _pieces(racing, tokens, (5, 17, 31, tokens.shape[1] - 53))
    assert torch.equal(whole, cut)


def test_the_earlier_passes_learn_through_their_own_evidence():
    racing = _stack(dtype=torch.float64, stack_loop=(0, 1, 2), stack_loop_exits=True).train()
    racing(_bytes())[..., 0].sum().backward()
    grads = {name: parameter.grad for name, parameter in racing.named_parameters() if parameter.grad is not None}
    assert any(grad.abs().sum() > 0 for grad in grads.values())
    racing.zero_grad()
    racing(_bytes())
    racing.exit_counts[0][..., 0].sum().backward()
    assert racing.output.weight.grad is not None and racing.output.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("settings", [dict(stack_loop_exits=True), dict(stack_loop=(0, 1, 1), stack_loop_exits=True),
                                      dict(stack_loop=(0, 1, 2), stack_loop_exits=True, stack_sleep=16),
                                      dict(stack_loop=(0, 1, 2), stack_loop_exits=1)])
def test_a_racing_loop_needs_an_awake_loop_of_two_passes(settings):
    with pytest.raises(ValueError, match="racing loop"):
        EventLM(EventLMConfig(**dict(_STACK, **settings)))


def test_the_halting_curve_takes_the_first_pass_that_is_sure():
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import loop_exits

    # Three bytes, two earlier passes and the top.
    bits = np.array([[1.0, 5.0, 2.0], [0.5, 4.0, 1.0], [0.2, 3.0, 0.1]])
    margins = np.array([[9.0, 0.1, 1.0], [9.0, 0.2, 5.0], [0.0, 0.0, 0.0]])
    (row,) = loop_exits.halting_curve(bits, margins, thresholds=(4.0,))
    # The first byte is decided at the first pass, the second never (the top), the third at the second pass.
    assert row["bits_per_byte"] == round((1.0 + 3.0 + 1.0) / 3, 6)
    assert row["mean_passes"] == round((1 + 3 + 2) / 3, 4) and row["decided_early"] == round(2 / 3, 4)


def test_the_top_pass_scores_as_the_stream_scorer_does():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_stream
    import loop_exits

    racing = _stack(stack_loop=(0, 1, 3), stack_loop_exits=True)
    data = torch.randint(0, 256, (600,), generator=torch.Generator().manual_seed(4))
    rows = eval_stream.stream_rows(data, 2, 64)
    bits, margins = loop_exits.pass_bits(racing, rows, 50, "cpu")
    assert bits.shape == margins.shape and bits.shape[0] == 3 and (margins >= 0).all()
    pieces, targets = eval_stream.score(racing, rows, 50, "cpu")
    assert abs(bits[-1].mean() - pieces.sum() / targets) < 1e-6
