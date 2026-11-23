"""The skip-gain report (experiments/skip_gains.py): what a stack's skip fibres learned, read from its weights."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from snnformer.event_lm import _SKIP_LIMIT, EventLM, EventLMConfig
from test_teaching_spikes import _STACK

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import skip_gains as tool  # noqa: E402
from skip_gains import skip_gains  # noqa: E402

_DEEP = dict(_STACK, depth=3, read_after=(0, 1, 2), granule_after=(0, 1, 2))


def _stack(**settings):
    torch.manual_seed(0)
    return EventLM(EventLMConfig(**dict(_DEEP, **settings))).eval()


def test_untrained_fibres_report_no_gain():
    found = skip_gains(_stack(stack_skips="level"))
    # cells0, read1, cells1, read2 and cells2 hear 1, 1, 2, 2 and 3 places: nine rows of 64 channels.
    assert found["skips"] == "level" and len(found["rows"]) == 9
    assert found["whole"] == {"gains": 9 * 64, "share_not_zero": 0.0, "mean_size_sixteenths": 0.0,
                              "largest_sixteenths": 0.0, "mean_stored_size_sixteenths": 0.0}
    assert set(found["by_part"]) == {"read", "cells"}
    assert all(reach == 1.0 for reach in found["mean_reach_of_a_channel"].values())


def test_a_row_names_its_part_and_where_its_fibres_come_from():
    model = _stack(stack_skips="parts")
    with torch.no_grad():
        # The tags of the third read hear the input at 3/16 on half of their channels and at -1/16 on the rest.
        gains = model.skips["read2"].view(3, 2, 64)
        gains[1, 0, :32] = 3 / 16
        gains[1, 0, 32:] = -1 / 16
        # A gain under half a sixteenth is still zero.
        gains[2, 1] = 0.4 / 16
    found = skip_gains(model)
    rows = {(row["level"], row["part"], row["from"]): row for row in found["rows"]}
    assert len(rows) == 6 + 3 * 3   # the cells' six rows, and three parts of two reads with one and two places
    tags = rows[2, "tags", "the input"]
    assert tags["levels_back"] == 2 and tags["stack_level"] == 2
    assert tags["share_not_zero"] == 1.0 and tags["largest_sixteenths"] == 3.0
    assert tags["mean_size_sixteenths"] == pytest.approx(2.0) and tags["mean_gain_sixteenths"] == pytest.approx(1.0)
    assert rows[2, "payloads", "after level 0"]["share_not_zero"] == 0.0
    # The payloads' gains are still zero, but the values they are rounded from are on their way.
    assert found["by_part"]["payloads"]["mean_size_sixteenths"] == 0.0
    assert found["by_part"]["payloads"]["mean_stored_size_sixteenths"] == pytest.approx(0.4 / 3)
    assert rows[2, "payloads", "after level 0"]["levels_back"] == 1
    assert rows[2, "cells", "after level 1"]["levels_back"] == 0
    assert found["by_part"]["tags"]["gains"] == 3 * 64
    assert found["by_part"]["tags"]["share_not_zero"] == pytest.approx(1 / 3)
    # Two levels back: the three parts of the third read and its cells, each from the input.
    assert found["by_levels_back"]["2"]["gains"] == 4 * 64
    assert found["by_levels_back"]["2"]["mean_size_sixteenths"] == pytest.approx(2 / 4)
    assert found["whole"]["share_not_zero"] == pytest.approx(64 / (15 * 64))
    assert found["mean_reach_of_a_channel"]["level 2 tags"] == pytest.approx(1 + 2 / 16)
    assert found["mean_reach_of_a_channel"]["level 2 probes"] == 1.0


def test_gains_are_reported_as_the_stack_takes_them():
    model = _stack(stack_skips="level")
    with torch.no_grad():
        # Three places at 2 spikes each are 96 sixteenths on a channel; the stack shrinks them to 32 in all.
        model.skips["cells2"].fill_(2.0)
    found = skip_gains(model)
    rows = [row for row in found["rows"] if row["level"] == 2 and row["part"] == "cells"]
    assert len(rows) == 3
    assert sum(row["mean_size_sixteenths"] for row in rows) == pytest.approx(_SKIP_LIMIT, abs=1.5)
    assert found["mean_reach_of_a_channel"]["level 2 cells"] == pytest.approx(3.0, abs=0.1)


def test_every_pass_of_a_loop_is_a_level_with_its_own_rows():
    model = _stack(stack_skips="level", depth=2, read_after=(0, 1), granule_after=(0, 1), stack_loop=(0, 1, 2))
    found = skip_gains(model)
    levels = {row["level"]: row["stack_level"] for row in found["rows"]}
    assert levels == {0: 0, 1: 1, 2: 0, 3: 1}
    # The fourth level that a byte passes is the second pass of the stack's level 1: its cells hear four places.
    assert sum(row["level"] == 3 and row["part"] == "cells" for row in found["rows"]) == 4


def test_a_stack_without_fibres_is_refused():
    with pytest.raises(ValueError, match="no skip fibres"):
        skip_gains(_stack())


def test_the_tool_reads_checkpoints_and_writes_a_record(tmp_path, monkeypatch, capsys):
    model = _stack(stack_skips="parts")
    with torch.no_grad():
        model.skips["read1"].fill_(1 / 16)
    run = tmp_path / "sk-parts"
    run.mkdir()
    torch.save({"task": "event_lm", "model_type": "EventLM", "cfg": asdict(model.cfg), "run_name": "a run",
                "args": {"lr": 1.2e-3, "steps": 24000}, "state_dict": model.state_dict()}, run / "model.pt.averaged")
    monkeypatch.setattr(sys, "argv", ["skip_gains.py", "--checkpoint", str(run / "model.pt.averaged"), "--json",
                                      str(tmp_path / "gains.json")])
    tool.main()
    record = json.loads((tmp_path / "gains.json").read_text())
    found = record["runs"]["sk-parts"]
    assert record["gain_unit"] == "1/16 of a spike"
    assert found["run_name"] == "a run" and found["peak_rate"] == 1.2e-3 and found["schedule_updates"] == 24000
    assert found["by_part"]["probes"]["share_not_zero"] == pytest.approx(1 / 3)
    assert "sk-parts" in capsys.readouterr().out
