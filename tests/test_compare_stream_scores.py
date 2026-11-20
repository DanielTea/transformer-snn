"""Comparing two scorings of the same streams piece by piece."""

import json
from pathlib import Path

import numpy as np
import pytest

_STREAMS, _BYTES = 3, 37


@pytest.fixture
def comparer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import compare_stream_scores

    return compare_stream_scores


def _pieces(bits, piece):
    """The bits of every byte summed over pieces of this many bytes, the last one shorter."""
    return np.add.reduceat(bits, np.arange(0, bits.shape[-1], piece), axis=-1)


def _stack_scoring(folder, name, bits, piece, **changed):
    """A record and the kept bits as eval_stream.py writes them for one checkpoint."""
    record = dict({"protocol": "unbroken_streams", "split": "validation", "streams": _STREAMS,
                   "unscored_lead_bytes": 0, "piece_bytes": piece,
                   "checkpoints": [{"checkpoint": f"{name}.pt", "targets": bits.size, "bpb": bits.sum() / bits.size}]},
                  **changed)
    (folder / f"{name}.json").write_text(json.dumps(record))
    np.save(folder / f"{name}.npy", _pieces(bits, piece)[None])
    return [str(folder / f"{name}.json"), str(folder / f"{name}.npy")]


def _reference_scoring(folder, name, rows, context):
    """A record and the kept bits as eval_taught_reference.py writes them for several rates."""
    record = {"protocol": "unbroken_streams_in_warmed_windows", "split": "validation", "streams": _STREAMS,
              "unscored_lead_bytes": 0, "context_bytes": context, "history_bytes": 8, "checkpoint": f"{name}.pt",
              "targets": rows[0][1].size,
              "fast_output_synapses": [{"rate": rate, "bpb": bits.sum() / bits.size} for rate, bits in rows]}
    (folder / f"{name}.json").write_text(json.dumps(record))
    np.save(folder / f"{name}.npy", np.stack([_pieces(bits, context) for _, bits in rows]))
    return [str(folder / f"{name}.json"), str(folder / f"{name}.npy")]


def _run(comparer, monkeypatch, folder, first, second, *more):
    out = folder / f"compared{len(list(folder.glob('compared*')))}.json"
    monkeypatch.setattr("sys.argv", ["compare_stream_scores.py", "--first", *first, "--second", *second,
                                     "--json", str(out), *more])
    comparer.main()
    return json.loads(out.read_text())


def test_a_stack_and_a_transformer_are_compared_at_the_longer_piece(comparer, monkeypatch, tmp_path):
    generator = np.random.default_rng(0)
    stack = generator.uniform(0, 3, (_STREAMS, _BYTES))
    frozen = stack + generator.normal(0.2, 0.5, stack.shape)
    taught = frozen - 0.1
    first = _stack_scoring(tmp_path, "stack", stack, 8)
    second = _reference_scoring(tmp_path, "transformer", [(0.0, frozen), (0.3, taught)], 4)
    result = _run(comparer, monkeypatch, tmp_path, first, second, "--second-row", "1")
    # Five pieces of 8 bytes a stream, the last of 5 bytes: the ten windows of 4 bytes are summed in pairs.
    assert (result["piece_bytes"], result["pieces"], result["targets"]) == (8, _STREAMS * 5, stack.size)
    assert result["first"]["bpb"] == pytest.approx(stack.sum() / stack.size)
    assert result["second"] == {"scoring": "transformer.pt at rate 0.3",
                                "bpb": pytest.approx(taught.sum() / stack.size)}
    difference = (taught.sum() - stack.sum()) / stack.size
    assert result["bpb_second_over_first"] == pytest.approx(difference)
    low, high = result["bpb_second_over_first_paired_95"]
    assert low < difference < high
    # The pieces are resampled together, so the interval is narrower than the spread of either scoring alone.
    alone = np.random.default_rng(1).integers(_STREAMS * 5, size=(3000, _STREAMS * 5))
    spread = np.percentile(_pieces(stack, 8).reshape(-1)[alone].sum(1) / stack.size, (2.5, 97.5))
    assert high - low < spread[1] - spread[0]


def test_a_scoring_differs_from_itself_by_nothing(comparer, monkeypatch, tmp_path):
    bits = np.random.default_rng(2).uniform(0, 3, (_STREAMS, _BYTES))
    scoring = _stack_scoring(tmp_path, "stack", bits, 8)
    result = _run(comparer, monkeypatch, tmp_path, scoring, scoring)
    assert result["bpb_second_over_first"] == 0
    assert result["bpb_second_over_first_paired_95"] == [0, 0]


def test_scorings_of_other_streams_or_pieces_that_do_not_line_up_are_refused(comparer, monkeypatch, tmp_path, capsys):
    bits = np.random.default_rng(3).uniform(0, 3, (_STREAMS, _BYTES))
    scoring = _stack_scoring(tmp_path, "stack", bits, 8)
    for name, piece, changed, said in (("lead", 8, {"unscored_lead_bytes": 16}, "differ in unscored_lead_bytes"),
                                       ("split", 8, {"split": "test"}, "differ in split"),
                                       ("short", 8, {"first_bytes_of_split": 100}, "differ in first_bytes_of_split"),
                                       ("piece", 6, {}, "do not line up")):
        other = _stack_scoring(tmp_path, name, bits, piece, **changed)
        with pytest.raises(SystemExit):
            _run(comparer, monkeypatch, tmp_path, scoring, other)
        assert said in capsys.readouterr().err
    fewer = _stack_scoring(tmp_path, "fewer", bits[:, :-1], 8)
    with pytest.raises(SystemExit):
        _run(comparer, monkeypatch, tmp_path, scoring, fewer)
    assert "targets" in capsys.readouterr().err
