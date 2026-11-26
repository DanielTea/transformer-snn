"""The sleep-gate diagnostic (experiments/sleep_gates.py): how many quarters of the stored cells sleep's gates keep."""
import sys
from pathlib import Path

import pytest
import torch

from snnformer.event_lm import _SLEEP_QUARTERS, EventLM, EventLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from sleep_gates import KINDS, sleep_gates  # noqa: E402

_LEVELS = (0, 1)
_WINDOW = 16
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def _stack(**options):
    torch.manual_seed(0)
    model = EventLM(EventLMConfig(**_STACK, stack_sleep=_WINDOW, **options))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model.eval()


def _bytes(length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(2))


@pytest.mark.parametrize("keep, quarters", [(0.95, 4), (0.5, 2), (0.3, 1)])
def test_deaf_gates_keep_the_quarters_of_their_rest(keep, quarters):
    found = sleep_gates(_stack(stack_sleep_keep=keep), _bytes(), 32, "cpu", skip=0)
    assert found
    for row in found.values():
        for kind in KINDS:
            assert row[kind]["cells"] > 0
            assert row[kind]["share_by_quarters_kept"][quarters] == 1.0
            assert row[kind]["mean_share_kept"] == pytest.approx(quarters / _SLEEP_QUARTERS)


def test_the_count_leaves_the_stream_as_it_was_and_unwraps_the_reads():
    model, tokens = _stack(), _bytes()
    with torch.no_grad():
        for read in dict.fromkeys(model.state_reads.values()):
            read.sleep_gate.copy_(torch.randn(read.sleep_gate.shape, generator=torch.Generator().manual_seed(3)))
        before, _ = model.stream(tokens)
    found = sleep_gates(model, tokens, 32, "cpu", skip=0)
    with torch.no_grad():
        after, _ = model.stream(tokens)
    assert torch.equal(before, after)
    assert all("_blend" not in read.__dict__ for read in model.state_reads.values())
    # Gates that hear the cells spread their quarters.
    assert any(max(row[kind]["share_by_quarters_kept"]) < 1.0 for row in found.values() for kind in KINDS)


def test_windows_in_the_lead_are_not_counted():
    found = sleep_gates(_stack(), _bytes(), 32, "cpu", skip=96)
    assert all(row[kind]["cells"] == 0 for row in found.values() for kind in KINDS)


def test_a_stack_that_does_not_sleep_is_refused():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="does not sleep"):
        sleep_gates(EventLM(EventLMConfig(**_STACK)).eval(), _bytes(), 32, "cpu", skip=0)
