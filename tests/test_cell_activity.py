"""The cell-activity diagnostic (experiments/cell_activity.py): how densely a stack's threshold cells fire."""
import sys
from pathlib import Path

import torch

from snnformer.event_lm import EventLM, EventLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from cell_activity import activity  # noqa: E402

_LEVELS = (0, 1)
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def _stack():
    torch.manual_seed(0)
    model = EventLM(EventLMConfig(**_STACK))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
            layer.receive.mul_(4)
    return model.eval()


def _bytes(length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(2))


def test_every_level_reports_shares_and_counts():
    levels = activity(_stack(), _bytes(), 32, torch.device("cpu"))
    assert len(levels) == 2
    for level in levels:
        assert level["cells"] == 96
        assert 0 < level["share_of_cells_firing_at_a_byte"] <= 1
        # Nine tenths of a level's spikes come from some of the cells that fire, and from no more than fire.
        assert 0 < level["share_of_cells_sending_nine_tenths"] <= level["share_of_cells_firing_at_a_byte"]
        spread = level["share_of_bytes_a_cell_fires_at"]
        assert 0 <= spread["p10"] <= spread["median"] <= spread["p90"] <= 1
        assert level["spikes_of_a_firing_cell"]["median"] >= 1
        assert 0 <= level["cells_that_never_fire"] <= 96


def test_pieces_do_not_change_the_measure_and_skipped_bytes_are_left_out():
    model, data = _stack(), _bytes()
    whole = activity(model, data, 96, torch.device("cpu"))
    pieces = activity(model, data, 16, torch.device("cpu"))
    for a, b in zip(whole, pieces, strict=True):
        assert a == b
    later = activity(model, data, 16, torch.device("cpu"), skip=40)
    alone = activity(model, data, 96, torch.device("cpu"), skip=40)
    assert later == alone
    assert later != whole


def test_nine_tenths_of_the_spikes_come_from_the_strongest_cells():
    model, data = _stack(), _bytes()
    counts = []
    layer = next(iter(model.granule_layers.values()))
    handle = layer.fire.register_forward_hook(lambda _module, _inputs, fired: counts.append(fired.detach()))
    try:
        with torch.no_grad():
            model.stream(data)
    finally:
        handle.remove()
    fired = torch.cat(counts, dim=1).flatten(0, 1).flatten(1).float()
    needed = 0
    for row in fired:
        largest, total, sent = row.sort(descending=True).values, row.sum(), 0.0
        for cells, spikes in enumerate(largest, start=1):
            sent += float(spikes)
            if total > 0 and sent >= 0.9 * float(total) - 1e-6:
                needed += cells
                break
    level = activity(model, data, 96, torch.device("cpu"))[0]
    assert level["share_of_cells_sending_nine_tenths"] == needed / fired.numel()


def test_a_cell_fires_from_half_a_threshold_of_charge_on():
    for level in activity(_stack(), _bytes(), 32, torch.device("cpu")):
        by_charge = level["share_of_cells_and_bytes_by_charge_in_thresholds"]
        assert list(by_charge) == ["below -8", "-8 to -4", "-4 to -2", "-2 to -1", "-1 to 0", "0 to 0.5",
                                   "0.5 and above"]
        assert abs(sum(by_charge.values()) - 1) < 1e-12
        # The cells with half a threshold of charge or more are the cells that fire.
        assert by_charge["0.5 and above"] == level["share_of_cells_firing_at_a_byte"]
        assert sum(by_charge[name] for name in ("below -8", "-8 to -4", "-4 to -2", "-2 to -1", "-1 to 0")) > 0
