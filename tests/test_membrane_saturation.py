"""The membrane-saturation diagnostic (experiments/membrane_saturation.py): how often neurons sit at their bound."""
import sys
from pathlib import Path

import torch

from snnformer.event_lm import EventLM, EventLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from membrane_saturation import kind_of, saturation  # noqa: E402

_LEVELS = (0, 1)
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def _stack(loud=1.0, **options):
    torch.manual_seed(0)
    model = EventLM(EventLMConfig(**_STACK, **options))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
            layer.receive.mul_(loud)
    return model.eval()


def _bytes(length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(2))


def test_names_sort_into_kinds_and_levels():
    assert kind_of("state_reads.3.fire") == ("reads", 3)
    assert kind_of("granule_layers.10.fire") == ("cells", 10)
    assert kind_of("granule_layers.2.emit") == ("stream after cells", 2)
    assert kind_of("output_fire")[0] == "other"


def test_shares_are_fractions_and_loud_cells_reach_the_bound():
    quiet = saturation(_stack(), _bytes(), 32, torch.device("cpu"))
    loud = saturation(_stack(loud=400.0), _bytes(), 32, torch.device("cpu"))
    for found in (quiet, loud):
        assert set(found) == {"by_kind", "by_level"}
        assert all(0 <= value <= 1 for value in found["by_kind"].values())
        assert {"cells 0", "cells 1", "reads 0", "reads 1"} <= set(found["by_level"])
    assert loud["by_kind"]["cells"] > quiet["by_kind"]["cells"]
    assert loud["by_kind"]["cells"] > 0


def test_lifted_cells_count_their_cap_and_pieces_do_not_change_the_measure():
    model = _stack(loud=400.0, pool_lifts_ceiling=True)
    whole = saturation(model, _bytes(), 96, torch.device("cpu"), skip=16)
    pieces = saturation(model, _bytes(), 16, torch.device("cpu"), skip=16)
    assert whole == pieces
    assert "cells" in whole["by_kind"] and 0 <= whole["by_kind"]["cells"] <= 1
