"""The gate diagnostic (experiments/gate_state.py): where a stack's pulses go and how far its gates stand open."""
import sys
from pathlib import Path

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig, _IntegrateFire, emitted_pulses

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from gate_state import gate_state, population_of  # noqa: E402

_LEVELS = (0, 1)
# The small stack of the stream-form tests.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)
_GATED = ("gate_open", "gate_wide_open", "share_when_open", "open_gates_by_quarter_share", "gate_steps_a_byte",
          "body_at_most", "mean_quanta")


def _stack(loud=1.0, gate_loud=1.0, **options):
    torch.manual_seed(0)
    model = EventLM(EventLMConfig(**_STACK, **options))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
            layer.receive.mul_(loud)
            if getattr(layer, "gate", None) is not None:
                layer.gate.mul_(gate_loud)
    return model.eval()


def _bytes(length=96, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(2))


def _measure(model, piece=32, skip=0):
    return gate_state(model, _bytes(), piece, torch.device("cpu"), skip=skip)


def test_names_sort_into_populations():
    assert population_of("state_reads.3.fire") == "reads"
    assert population_of("state_reads.3.emit") == "stream after reads"
    assert population_of("granule_layers.10.fire") == "cells"
    assert population_of("granule_layers.10.open") == "gate dendrites"
    assert population_of("granule_layers.2.emit") == "stream after cells"
    assert population_of("granule_layers.2.pool") == population_of("state_reads.0.pool") == "pools"
    assert population_of("output_pool") == "pools"
    assert population_of("output_emit") == "output"
    # An unknown population keeps its name, with its level folded so that the levels add up.
    assert population_of("new_part.4.fire") == population_of("new_part.7.fire") == "new_part.N.fire"


def test_pulses_are_those_of_the_trainers_budget():
    model = _stack(granule_gate=4, granule_shunt=True)
    # The trainer's count: every firing population's pulses and what the model adds itself, in one list.
    counted = []
    handles = [module.register_forward_hook(lambda module, _args, fired: counted.append(emitted_pulses(module, fired)))
               for module in model.modules() if isinstance(module, _IntegrateFire)]
    with torch.no_grad():
        model.stream(_bytes(), memory_pulses=counted)
    for handle in handles:
        handle.remove()
    found = _measure(model, piece=96)
    assert found["pulses_a_byte"] == pytest.approx(float(torch.stack(counted).sum()) / _bytes().numel())
    assert found["pulses_a_byte"] == pytest.approx(sum(found["pulses_by_population"].values()))
    assert list(found["pulses_by_population"].values()) == sorted(found["pulses_by_population"].values(), reverse=True)
    # A race read counts its racing cells' spikes and its read pulses itself, and the measure names them.
    assert found["pulses_by_population"]["racing cells"] > 0
    assert found["pulses_by_population"]["read pulses"] > 0
    assert "within reads" not in found["pulses_by_population"]


def test_plain_cells_report_their_firing_and_no_gate():
    found = _measure(_stack())
    assert set(found["cells_by_level"]) == {"0", "1"}
    for row in found["cells_by_level"].values():
        assert set(row) == {"firing", "mean_count", "spikes_a_byte"}
        assert 0 <= row["firing"] <= 1
        # 96 cells a level: the spikes a byte are the mean count of all of them.
        assert row["spikes_a_byte"] == pytest.approx(96 * row["mean_count"])
    assert "gate dendrites" not in found["pulses_by_population"]


@pytest.mark.parametrize("options", [dict(), dict(granule_shunt=True), dict(granule_ramp=16),
                                     dict(granule_ramp=16, pool_lifts_ceiling=True),
                                     dict(granule_shunt=True, pool_lifts_ceiling=True)])
def test_gated_cells_report_their_gates(options):
    found = _measure(_stack(granule_gate=4, **options))
    for row in found["cells_by_level"].values():
        assert set(row) == {"firing", "mean_count", "spikes_a_byte", *_GATED}
        assert all(0 <= row[key] <= 1 for key in ("firing", "gate_open", "gate_wide_open", "body_at_most"))
        assert row["gate_wide_open"] <= row["gate_open"]
        # A body fires only where its gate is open.
        assert row["firing"] <= row["gate_open"]
        assert row["mean_quanta"] >= 1
        # Every open gate lets in a share that lies in one of the four quarters.
        assert len(row["open_gates_by_quarter_share"]) == 4
        assert sum(row["open_gates_by_quarter_share"]) == pytest.approx(1)
        if not options.get("granule_shunt") and not options.get("granule_ramp"):
            continue
        # A shunting gate and a gate that opens in proportion never let in more than all of the charge.
        assert 0 < row["share_when_open"] <= 1
    if options.get("granule_shunt"):
        # A shunting gate with k steps lets in k / (k + 4): never all of it, and a fifth with its first step.
        assert all(row["gate_wide_open"] == 0 for row in found["cells_by_level"].values())
    if not options:
        # A multiplying gate lets in a quarter with each of its steps: its fourth step and all after are wide open,
        # and they are the open gates of the last quarter.
        for row in found["cells_by_level"].values():
            assert row["open_gates_by_quarter_share"][3] * row["gate_open"] == pytest.approx(row["gate_wide_open"])


def test_a_firing_gate_spends_pulses_and_an_inner_gate_spends_none():
    firing = _measure(_stack(granule_gate=4, granule_shunt=True))
    inner = _measure(_stack(granule_gate=4, granule_shunt=True, granule_inner_gate=True))
    steps = sum(row["gate_steps_a_byte"] for row in firing["cells_by_level"].values())
    assert steps > 0
    assert firing["pulses_by_population"]["gate dendrites"] == pytest.approx(steps)
    assert inner["pulses_by_population"]["gate dendrites"] == 0
    # The inner gate takes the same steps. Only the budget's count of them differs at the first level; the levels
    # after hear the same spikes, so every other population fires as before.
    assert sum(row["gate_steps_a_byte"] for row in inner["cells_by_level"].values()) == pytest.approx(steps)
    assert inner["pulses_a_byte"] == pytest.approx(firing["pulses_a_byte"] - steps)


def test_a_louder_gate_dendrite_opens_a_proportional_gate_wider():
    quiet = _measure(_stack(granule_gate=4, granule_ramp=16, pool_lifts_ceiling=True))
    loud = _measure(_stack(gate_loud=30.0, granule_gate=4, granule_ramp=16, pool_lifts_ceiling=True))
    for level in ("0", "1"):
        assert loud["cells_by_level"][level]["share_when_open"] > quiet["cells_by_level"][level]["share_when_open"]
        # The louder gate's open states move from the first quarter towards the last.
        assert (loud["cells_by_level"][level]["open_gates_by_quarter_share"][3]
                > quiet["cells_by_level"][level]["open_gates_by_quarter_share"][3])
    assert loud["cells_by_level"]["0"]["gate_wide_open"] > quiet["cells_by_level"]["0"]["gate_wide_open"]


def test_the_cut_of_the_stream_and_the_lead_do_not_change_the_measure():
    model = _stack(granule_gate=4, granule_ramp=16, granule_inner_gate=True, pool_lifts_ceiling=True)
    whole = gate_state(model, _bytes(), 96, torch.device("cpu"), skip=16)
    pieces = gate_state(model, _bytes(), 16, torch.device("cpu"), skip=16)
    odd = gate_state(model, _bytes(), 23, torch.device("cpu"), skip=16)
    assert whole["cells_by_level"] == pieces["cells_by_level"] == odd["cells_by_level"]
    for other in (pieces, odd):
        assert other["pulses_a_byte"] == pytest.approx(whole["pulses_a_byte"])
        assert other["pulses_by_population"].keys() == whole["pulses_by_population"].keys()
    # The lead bytes are left out: the measure over the bytes after them is not that over all bytes.
    assert gate_state(model, _bytes(), 96, torch.device("cpu"))["pulses_a_byte"] != whole["pulses_a_byte"]
