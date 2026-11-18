"""A training run that is stopped and continued ends where an unbroken run ends."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


TINY = """--variant event_correction_code --no-carry --dim 32 --hidden 16 --depth 2 --memory-heads 2
--memory-slots 4 --memory-width 8 --spike-resolution 16 --pulse-budget 4000 --code-loss-weight 0.25
--read-after 0 --read-width 16 --read-window 16 --read-mode race --read-budget 16 --read-beats 2
--granule-after 0 --granule-clusters 2 --granule-cells 8 --granule-fan 8 --granule-budget 4
--granule-delays 2 --granule-beats 2 --granule-clock learned --rotated-adam 2 --weight-average 0.9
--ctx 16 --bs 2 --steps 8 --warmup 2 --lr 0.003 --seed 3 --eval-every 2 --eval-bs 2 --eval-batches 1
--final-eval-batches 1 --validation-only --save-every 2 --resume""".split()


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))

    def run(folder, *extra):
        monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *TINY, "--checkpoint", str(folder / "model.pt"),
                                          "--results-dir", str(folder / "results"), *extra])
        train_event_lm.main()

    return run


def record(folder):
    (path,) = (folder / "results").glob("*.json")
    return json.loads(path.read_text())


def test_a_stopped_run_continues_to_the_weights_of_an_unbroken_run(trainer, tmp_path):
    whole, broken = tmp_path / "whole", tmp_path / "broken"
    trainer(whole)
    trainer(broken, "--stop-at", "4")
    assert not (broken / "model.pt").exists() and (broken / "model.pt.resume").exists()
    trainer(broken, "--stop-at", "6")
    trainer(broken)

    for name in ("model.pt", "model.pt.averaged"):
        one = torch.load(whole / name, weights_only=True)["state_dict"]
        other = torch.load(broken / name, weights_only=True)["state_dict"]
        assert one.keys() == other.keys()
        assert all(torch.equal(one[key], other[key]) for key in one), name
    one, other = record(whole), record(broken)
    assert other["resumed_at_steps"] == [4, 6] and "resumed_at_steps" not in one
    assert [point["step"] for point in other["curve"]] == [2, 4, 6, 8]
    for key in ("val_bpb", "averaged_val_bpb", "train_bpb", "spike_price", "train_pulses_per_byte"):
        assert [point[key] for point in one["curve"]] == [point[key] for point in other["curve"]], key
    assert not (broken / "model.pt.resume").exists() and not (whole / "model.pt.resume").exists()


def test_a_run_continues_only_with_the_settings_it_was_started_with(trainer, tmp_path, capsys):
    trainer(tmp_path, "--stop-at", "4")
    with pytest.raises(SystemExit):
        trainer(tmp_path, "--lr", "0.001")
    assert "the run differs in lr" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        trainer(tmp_path, "--stop-at", "4")
    assert "already at update 4" in capsys.readouterr().err


def test_stopping_needs_a_saved_state_to_continue_from(trainer, tmp_path, capsys):
    with pytest.raises(SystemExit):
        trainer(tmp_path, "--stop-at", "3")
    assert "--stop-at needs --resume" in capsys.readouterr().err


def test_a_trained_decoder_may_gain_word_ages_and_nothing_else(trainer, tmp_path, capsys):
    parent = tmp_path / "parent"
    trainer(parent, "--read-beats", "0")
    trainer(tmp_path / "aged", "--load-weights", str(parent / "model.pt"))
    loaded = record(tmp_path / "aged")["args"]["load_weights"]
    assert loaded["checkpoint"] == str(parent / "model.pt") and len(loaded["sha256"]) == 64
    with pytest.raises(SystemExit):
        trainer(tmp_path / "wider", "--load-weights", str(parent / "model.pt"), "--read-width", "32")
    assert "the decoder differs in read_width" in capsys.readouterr().err
