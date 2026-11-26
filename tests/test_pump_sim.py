"""The pump simulator (experiments/pump_sim.py): its lags, its bytes, the signatures of its faults and its corpus."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from pump_sim import CHANNELS, RANGES, Fault, Pump, Run, decode, encode, lag, simulate, write_corpus  # noqa: E402
from train_event_lm import load_corpus  # noqa: E402

HOUR = 3600


def _channel(values, name):
    return values[:, CHANNELS.index(name)]


def test_the_lag_is_the_step_by_step_recurrence():
    target = np.random.default_rng(0).normal(50, 10, 5000)
    value, slow = target[0], []
    for aim in target:
        value += (aim - value) / 900
        slow.append(value)
    np.testing.assert_allclose(lag(target, 900, block=777), slow, rtol=1e-9)


def test_a_byte_holds_a_value_to_half_a_step_of_its_range():
    values = simulate(Run(Pump(), 600, seed=1))
    step = np.array([(RANGES[c][1] - RANGES[c][0]) / 255 for c in CHANNELS])
    assert (np.abs(decode(encode(values)) - values.clip([RANGES[c][0] for c in CHANNELS],
                                                         [RANGES[c][1] for c in CHANNELS])) <= step / 2 + 1e-9).all()


def test_a_healthy_pump_obeys_its_physics():
    values = simulate(Run(Pump(), 2 * HOUR, seed=2))
    running = _channel(values, "speed_rpm") > 0.7 * Pump().n0
    assert running.mean() > 0.5
    flow, suction, discharge = (_channel(values, name)[running] for name in ("flow_m3h", "suction_bar", "discharge_bar"))
    assert (flow > 0).all() and (discharge > suction).all()
    stopped = _channel(values, "speed_rpm") < 100
    assert (_channel(values, "flow_m3h")[stopped] < 5).all()


def _before_after(kind, name, severity=1.0, channel=None, seconds=4 * HOUR):
    run = Run(Pump(), seconds, [Fault(kind, seconds // 2, 60, severity, channel)], seed=3)
    healthy = Run(Pump(), seconds, [], seed=3)
    late = slice(seconds - HOUR // 2, seconds)
    return _channel(simulate(healthy), name)[late], _channel(simulate(run), name)[late]


@pytest.mark.parametrize("kind, name, direction", [
    ("cavitation", "flow_m3h", -1), ("cavitation", "vibration_mms", 1), ("cavitation", "suction_bar", -1),
    ("clogging", "suction_bar", -1), ("bearing_wear", "vibration_mms", 1), ("bearing_wear", "bearing_c", 1),
    ("imbalance", "vibration_mms", 1), ("misalignment", "bearing_c", 1), ("valve_stuck", "flow_m3h", -1),
    ("valve_stuck", "discharge_bar", 1), ("seal_leak", "flow_m3h", -1)])
def test_every_fault_leaves_its_signature(kind, name, direction):
    healthy, faulty = _before_after(kind, name)
    assert direction * (faulty.mean() - healthy.mean()) > 0.02 * (RANGES[name][1] - RANGES[name][0])


def test_sensor_faults_change_the_reading_and_not_the_pump():
    healthy, drifted = _before_after("sensor_drift", "flow_m3h", channel="flow_m3h")
    assert drifted.mean() - healthy.mean() > 0.2 * (RANGES["flow_m3h"][1] - RANGES["flow_m3h"][0])
    _, frozen = _before_after("sensor_freeze", "discharge_bar", channel="discharge_bar")
    assert np.ptp(frozen) == 0
    healthy_current, current = _before_after("sensor_drift", "current_a", channel="flow_m3h")
    np.testing.assert_allclose(current, healthy_current)


def test_the_corpus_is_eight_bytes_a_second_split_by_pumps(tmp_path):
    out = tmp_path / "pumps.bin"
    sidecar = write_corpus(out, pumps=6, seconds=300, seed=4, shares=(0.5, 0.25, 0.25))
    train, validation, test = load_corpus(out)
    assert len(train) + len(validation) + len(test) == 6 * 300 * len(CHANNELS)
    assert sidecar["bytes_per_second"] == 8 and len(train) == 3 * 300 * 8
    truth = json.loads(Path(f"{out}.truth.json").read_text())
    assert [row["split"] for row in truth] == ["train"] * 3 + ["validation"] * 2 + ["test"]
    assert all(row["bytes"] == 300 * 8 for row in truth)


def test_the_telemetry_scorer_counts_ranks_and_windows_right():
    import telemetry_eval
    import torch

    from snnformer.event_lm import EventLM, EventLMConfig
    from test_hop_tasks import _STACK

    rng = np.random.default_rng(5)
    scores = rng.integers(0, 6, 300).astype(float)
    labels = rng.uniform(size=300) < 0.3
    pairs = [(p > n) + 0.5 * (p == n) for p in scores[labels] for n in scores[~labels]]
    assert abs(telemetry_eval.auc(scores, labels) - np.mean(pairs)) < 1e-12
    assert telemetry_eval.auc(scores, np.zeros(300, bool)) is None
    x = rng.normal(size=(2, 50))
    expected = np.stack([[x[r, max(0, i - 6):i + 1].mean() for i in range(50)] for r in range(2)])
    np.testing.assert_allclose(telemetry_eval.running_mean(x, 7), expected)
    codes = rng.integers(0, 256, (3, 40, len(CHANNELS))).astype(np.uint8)
    assert telemetry_eval.zscore_detector(codes, 5).shape == (3, 40)
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    bits = telemetry_eval.surprise(stack, codes, "cpu", piece=64)
    assert bits.shape == (3, 40) and (bits > 0).all()
