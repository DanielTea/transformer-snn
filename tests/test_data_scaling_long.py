"""The data-scaling analysis (experiments/data_scaling_long.py): curves at three rates, reused runs, best rates."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from data_scaling_long import collect, figure, summarize  # noqa: E402


def _law(floor, scale, beta, thousands):
    return floor + scale * (thousands * 1000 * 4096) ** -beta


def _write(folder, name, bpb, transformer=False):
    record = {"fast_output_synapses": [{"rate": 0.0, "bpb": bpb}]} if transformer else {"checkpoints": [{"bpb": bpb}]}
    (folder / f"{name}.json").write_text(json.dumps(record))


def _scores(folder):
    # Each model is best at 1.2e-3; the plain stack's gap to the Transformer shrinks with data, the delayed stack's grows.
    models = {"tf": (1.05, 60, 0.21), "st": (1.20, 25, 0.19), "pl": (1.05, 120, 0.235)}
    offsets = {"3": 0.04, "6": 0.015, "12": 0.0}
    for family, (floor, scale, beta) in models.items():
        for rate, extra in offsets.items():
            if family == "pl" and rate == "6":
                continue
            for size in (6, 12, 24, 48, 96):
                for seed in (1, 2):
                    value = _law(floor, scale, beta, size) + extra + 0.001 * seed
                    if size == 24:
                        prefix = {"tf": "tf24", "st": "bed24", "pl": "pl24"}[family]
                        name = f"{prefix}-s{seed}" if rate == "3" else f"{prefix}-lr{rate}-s{seed}"
                    else:
                        name = f"ds-{family}{rate}-{size}k-s{seed}"
                    _write(folder, name, value, transformer=family == "tf")


def test_curves_at_every_rate_and_the_reused_runs_are_read(tmp_path):
    _scores(tmp_path)
    points = collect(tmp_path)
    assert set(points) == {"tf3", "tf6", "tf12", "st3", "st6", "st12", "pl3", "pl12"}
    assert all(set(curve) == {6, 12, 24, 48, 96} for curve in points.values())
    assert all(set(curve[24]) == {1, 2} for curve in points.values())


def test_each_model_is_read_at_its_best_rate_and_the_slopes_have_their_signs(tmp_path):
    _scores(tmp_path)
    summary, rates = summarize(collect(tmp_path))
    assert rates == {"tf": "12", "st": "12", "pl": "12"}
    assert summary["gaps"]["st12"]["against"] == "tf12"
    assert summary["gaps"]["st12"]["slope"]["per_doubling"] > 0
    assert summary["gaps"]["pl12"]["slope"]["per_doubling"] < 0
    assert summary["gaps"]["pl12"]["slope"]["scales_the_other_way"]
    figure(collect(tmp_path), rates, summary, tmp_path)
    assert (tmp_path / "event_lm_data_scaling_long.png").is_file()


def test_a_transformer_best_at_2_4e_3_is_read_at_that_rate(tmp_path):
    _scores(tmp_path)
    for size in (6, 12, 24, 48, 96):
        for seed in (1, 2):
            value = _law(1.05, 60, 0.21, size) - 0.006 + 0.001 * seed
            name = f"tf24-lr24-s{seed}" if size == 24 else f"ds-tf24-{size}k-s{seed}"
            _write(tmp_path, name, value, transformer=True)
    points = collect(tmp_path)
    assert set(points["tf24"]) == {6, 12, 24, 48, 96}
    summary, rates = summarize(points)
    assert rates["tf"] == "24" and rates["st"] == "12"
    assert summary["gaps"]["st12"]["against"] == "tf24"


def test_one_rate_alone_falls_back_to_the_old_rate(tmp_path):
    _write(tmp_path, "tf24-lr12-s1", 1.44, transformer=True)
    _write(tmp_path, "bed24-s1", 1.55)
    _, rates = summarize(collect(tmp_path))
    assert rates["tf"] == "3" and rates["st"] == "3"
