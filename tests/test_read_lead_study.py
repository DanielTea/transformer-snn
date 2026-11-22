"""The shared reading of the test bed's lead studies (experiments/read_lead_study.py)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from read_lead_study import read  # noqa: E402


def _scores(folder, values):
    for name, bpb in values.items():
        (folder / f"{name}.json").write_text(json.dumps({"checkpoints": [{"bpb": bpb}]}))


def test_a_lead_that_grows_with_data_passes(tmp_path):
    _scores(tmp_path, {"b24-s1": 1.50, "b24-s2": 1.52, "b6-s1": 1.80, "b6-s2": 1.82,
                       "x24-s1": 1.48, "x24-s2": 1.505, "x6-s1": 1.795, "x6-s2": 1.815})
    verdict = read(tmp_path, ["b24-s1", "b24-s2"], ["b6-s1", "b6-s2"], {"x": ["x24-s1", "x24-s2"]},
                   {"x": ["x6-s1", "x6-s2"]}, 0.010, 0.005, None)["x"]
    assert verdict["a_lead"] and verdict["b_growth"] and verdict["verdict"].startswith("pass")
    assert abs(verdict["long_mean_lead"] - 0.0175) < 1e-9 and abs(verdict["growth"] - 0.0125) < 1e-9


def test_a_constant_gain_is_not_a_fix(tmp_path):
    _scores(tmp_path, {"b24-s1": 1.50, "b24-s2": 1.52, "b6-s1": 1.80, "b6-s2": 1.82,
                       "x24-s1": 1.485, "x24-s2": 1.505, "x6-s1": 1.785, "x6-s2": 1.805})
    verdict = read(tmp_path, ["b24-s1", "b24-s2"], ["b6-s1", "b6-s2"], {"x": ["x24-s1", "x24-s2"]},
                   {"x": ["x6-s1", "x6-s2"]}, 0.010, 0.005, None)["x"]
    assert verdict["a_lead"] and not verdict["b_growth"]
    assert verdict["verdict"] == "a constant gain, not a fix for late learning"


def test_every_seed_must_lead_unless_fewer_wins_are_asked(tmp_path):
    _scores(tmp_path, {f"b-s{s}": 1.5 for s in range(1, 5)} | {"x-s1": 1.46, "x-s2": 1.48, "x-s3": 1.49, "x-s4": 1.505})
    base, arm = [f"b-s{s}" for s in range(1, 5)], [f"x-s{s}" for s in range(1, 5)]
    assert read(tmp_path, base, [], {"x": arm}, {}, 0.010, 0.005, None)["x"]["verdict"] == "fail"
    assert read(tmp_path, base, [], {"x": arm}, {}, 0.010, 0.005, 3)["x"]["verdict"] == "pass"


def test_missing_scores_say_so(tmp_path):
    _scores(tmp_path, {"b24-s1": 1.50, "b24-s2": 1.52, "x24-s1": 1.48})
    verdict = read(tmp_path, ["b24-s1", "b24-s2"], [], {"x": ["x24-s1", "x24-s2"]}, {}, 0.010, 0.005, None)["x"]
    assert verdict["verdict"].startswith("incomplete") and "x24-s2" in verdict["verdict"]
