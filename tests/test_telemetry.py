"""Telemetry as byte streams: levels, encodings, and a reader that carries one capsule per stream."""

import numpy as np
import pytest
import torch

from snnformer.telemetry import Quantizer, StreamReader, decode, encode, sample_width
from test_teaching_spikes import _TAUGHT, _stack


def test_the_quantizer_spans_normal_values_and_clips_the_rest():
    normal = np.linspace(10.0, 20.0, 1001)
    quantizer = Quantizer.fit(normal, 100)
    levels = quantizer(np.array([10.0, 15.0, 20.0, -1e9, 1e9, np.nan]))
    # Normal values sit inside the range, with room beyond them; far values are clipped to the ends.
    assert 0 < levels[0] < levels[1] < levels[2] < 99
    assert levels[3] == 0 and levels[4] == 99 and levels[5] == 0
    assert abs(quantizer.value(quantizer(np.array([15.0])))[0] - 15.0) < (quantizer.high - quantizer.low) / 99


def test_a_constant_stretch_still_gives_a_range():
    quantizer = Quantizer.fit(np.full(50, 3.0), 10)
    assert quantizer.high > quantizer.low and quantizer(np.array([3.0]))[0] in range(10)


@pytest.mark.parametrize("encoding,levels,width", [("digits", 100, 3), ("digits", 1000, 4), ("printable", 94, 1)])
def test_every_encoding_writes_fixed_widths_that_decode(encoding, levels, width):
    values = np.arange(levels).reshape(2, -1) if levels % 2 == 0 else np.arange(levels)[None]
    written = encode(values, encoding, levels)
    assert sample_width(encoding, levels) == width and written.shape == values.shape + (width,)
    assert np.array_equal(decode(written, encoding), values)
    # Every byte is one that text holds: digits and a newline, or printable characters.
    assert ((written >= 33) & (written <= 126) | (written == 10)).all()


def test_levels_outside_the_range_are_refused():
    with pytest.raises(ValueError):
        encode(np.array([100]), "digits", 100)
    with pytest.raises(ValueError):
        sample_width("printable", 200)


def _samples(streams=3, count=60, seed=4):
    levels = torch.randint(0, 100, (streams, count), generator=torch.Generator().manual_seed(seed)).numpy()
    return encode(levels, "digits", 100)


@pytest.mark.parametrize("design", [{}, _TAUGHT])
@pytest.mark.parametrize("cuts", [(60,), (1,) * 60, (7, 13, 40), (59, 1)])
def test_a_stream_read_in_pieces_is_scored_as_one_pass(design, cuts):
    model, samples = _stack(**design), _samples()
    whole = StreamReader(model, 3).read(samples)
    reader, parts, start = StreamReader(model, 3), [], 0
    for cut in cuts:
        parts.append(reader.read(samples[:, start:start + cut]))
        start += cut
    assert np.array_equal(np.concatenate(parts, 1), whole) and reader.samples == 60
    # The first byte of all is predicted from nothing: eight bits; the sample's other two bytes cost something too.
    assert (whole[:, 0] > 8).all() and np.isfinite(whole).all()


def test_a_saved_capsule_resumes_the_same_stream(tmp_path):
    model, samples = _stack(**_TAUGHT), _samples()
    whole = StreamReader(model, 3).read(samples)
    reader = StreamReader(model, 3)
    first = reader.read(samples[:, :25])
    reader.save(tmp_path / "capsules.pt")
    resumed = StreamReader.resume(model, tmp_path / "capsules.pt")
    assert resumed.samples == 25
    assert np.array_equal(np.concatenate([first, resumed.read(samples[:, 25:])], 1), whole)


def test_every_stream_is_read_by_itself():
    model, samples = _stack(**_TAUGHT), _samples()
    together = StreamReader(model, 3).read(samples)
    alone = StreamReader(model, 1).read(samples[1:2])
    assert np.allclose(together[1:2], alone, rtol=0, atol=1e-9)


def test_the_reader_refuses_samples_of_another_shape():
    reader = StreamReader(_stack(), 2)
    with pytest.raises(ValueError):
        reader.read(np.zeros((3, 4, 3), dtype=np.int64))


@pytest.fixture
def evaluator(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import telemetry_evaluate

    return telemetry_evaluate


def test_the_auc_counts_ties_half_and_needs_both_classes(evaluator):
    labels = np.array([False, False, True, True])
    assert evaluator.roc_auc(np.array([0.0, 1.0, 2.0, 3.0]), labels) == 1.0
    assert evaluator.roc_auc(np.array([3.0, 2.0, 1.0, 0.0]), labels) == 0.0
    assert evaluator.roc_auc(np.ones(4), labels) == 0.5
    assert evaluator.roc_auc(np.ones(4), np.zeros(4, dtype=bool)) is None


def test_anomalies_are_runs_of_labels_and_alarms_are_counted_once(evaluator):
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=bool)
    assert evaluator.segments(labels) == [(2, 4), (5, 6)]
    # Calibration: the first 4 samples score 0; after them an alarm opens in the second anomaly and one false alarm.
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0], dtype=bool)
    score = np.array([0, 0, 0, 0, 0, 0, 5, 0, 5, 5], dtype=float)
    result = evaluator.measure(score, labels, calibration=4, warm=0)
    assert result["caught"] == 1 and result["delays"] == [1] and result["anomalies"] == 1
    assert result["precision"] == pytest.approx(1 / 3) and result["recall"] == pytest.approx(1 / 2)
    assert result["false_alarms_per_1000_normal"] == pytest.approx(1000 / 4)


def test_the_baselines_rise_at_a_jump(evaluator):
    values = np.concatenate([np.sin(np.arange(300) / 5.0), 20 + np.sin(np.arange(300, 320) / 5.0)])[:, None]
    for scores in (evaluator.rolling_zscore(values, 50), evaluator.ewma_residuals(values),
                   evaluator.ar_residuals(values, 4, 200)):
        assert scores.shape == (1, 320) and scores[0, 300] > 10 * np.median(scores[0, 100:300])


def test_the_monitor_flags_a_stream_against_its_own_normal_and_clears_it():
    from snnformer.telemetry import SurpriseMonitor

    rng = np.random.default_rng(0)
    # Two streams with different normals: the second is always more surprising, and that alone is no event.
    normal = np.stack([rng.normal(3, 0.3, 400), rng.normal(9, 1.0, 400)])
    monitor = SurpriseMonitor(2, span=5)
    monitor.calibrate(normal, warm=20)
    quiet = monitor.update(np.stack([rng.normal(3, 0.3, 200), rng.normal(9, 1.0, 200)]))
    assert len(quiet) <= 2
    # The first stream's surprise jumps for a while, then returns.
    jump = np.stack([np.concatenate([rng.normal(8, 0.3, 30), rng.normal(3, 0.3, 60)]), rng.normal(9, 1.0, 90)])
    events = [event for event in monitor.update(jump) if event.stream == 0]
    kinds = [event.kind for event in events]
    assert kinds[:2] == ["watch", "alarm"] and kinds[-1] == "clear"
    assert 600 <= events[0].sample < 610


def test_a_monitor_snapshot_goes_on_where_it_stopped():
    from snnformer.telemetry import SurpriseMonitor

    rng = np.random.default_rng(1)
    normal, later = rng.normal(3, 0.3, (2, 300)), np.concatenate([rng.normal(3, 0.3, (2, 50)),
                                                                   rng.normal(7, 0.3, (2, 50))], 1)
    whole = SurpriseMonitor(2)
    whole.calibrate(normal)
    expected = whole.update(later)
    first = SurpriseMonitor(2)
    first.calibrate(normal)
    events = first.update(later[:, :40])
    resumed = SurpriseMonitor.restore(first.snapshot())
    assert events + resumed.update(later[:, 40:]) == expected
