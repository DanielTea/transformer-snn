"""Beyond its window a state read forgets by a power of the age, at a slope that every head learns."""

import math
from pathlib import Path
import sys

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig
from test_stream_form import _STACK, _bytes, _pieces, _stack, _train, _trainer


def _slopes(model, *values):
    with torch.no_grad():
        for read in model.state_reads.values():
            read.forget.copy_(torch.tensor(values))
    return model


def test_a_slope_at_zero_changes_nothing():
    tokens = _bytes()
    plain, forgetting = _stack(read_reach=96).eval(), _stack(read_reach=96, read_forget=True).eval()
    # The slopes are the only new parameters, one for every head of every read, and they start at zero.
    assert set(forgetting.state_dict()) - set(plain.state_dict()) == {"state_reads.0.forget", "state_reads.1.forget"}
    assert all(torch.equal(read.forget, torch.zeros(4)) for read in forgetting.state_reads.values())
    with torch.no_grad():
        assert torch.equal(forgetting(tokens), plain(tokens))
        assert torch.equal(_pieces(forgetting, tokens, (47, 1, 49, 63))[0], plain(tokens))


def test_a_far_cell_takes_the_slope_for_every_doubling_of_its_age():
    model = _slopes(_stack(read_reach=96, read_forget=True, read_forget_rate=0.5), 0.5, -1.0, 0.0, 2.0)
    read = model.state_reads["0"]
    aged = read.aged().detach()
    assert aged.shape == (4, 96) and torch.equal(aged[:, :48], read.age.detach())
    ages = torch.arange(48, 96, dtype=aged.dtype)
    expected = read.age.detach()[:, -1:] + 0.5 * torch.tensor([[0.5], [-1.0], [0.0], [2.0]]) * torch.log2(ages / 47)
    torch.testing.assert_close(aged[:, 48:], expected, rtol=0, atol=1e-6)
    # Twice as old as the window's oldest cell is one doubling: the slope times the rate, once.
    torch.testing.assert_close(aged[1, 94] - aged[1, 47], torch.tensor(-0.5), rtol=0, atol=1e-6)


def test_the_slope_learns_only_where_cells_lie_beyond_the_window():
    tokens = _bytes()
    model = _slopes(_stack(read_reach=96, read_forget=True), 0.25, -0.25, 0.5, -0.5)
    model.train()
    model(tokens).float().square().mean().backward()
    assert all(read.forget.grad is not None and read.forget.grad.abs().sum() > 0
               for read in model.state_reads.values())
    # Where a read reaches no further than its window, no cell is beyond it and the slope plays no part.
    near = _stack(read_forget=True)
    near.train()
    near(tokens).float().square().mean().backward()
    assert all(read.forget.grad is None for read in near.state_reads.values())


@pytest.mark.parametrize("cuts", [(160,), (47, 1, 49, 63), (7, 13, 100, 40), (1, 2, 3, 154)])
def test_a_forgetting_stream_cut_anywhere_fires_the_spikes_of_one_pass(cuts):
    tokens = _bytes()
    model = _slopes(_stack(read_reach=96, read_forget=True), 0.4, -0.7, 0.1, -1.3).eval()
    with torch.no_grad():
        whole = model(tokens)
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
        # The slopes change what the far cells fire.
        assert not torch.equal(whole, _stack(read_reach=96).eval()(tokens))


def test_a_rate_of_zero_holds_the_slope_at_nothing():
    tokens = _bytes()
    held = _slopes(_stack(read_reach=96, read_forget=True, read_forget_rate=0.0), 0.4, -0.7, 0.1, -1.3).eval()
    with torch.no_grad():
        assert torch.equal(held(tokens), _stack(read_reach=96).eval()(tokens))
    held.train()
    held(tokens).float().square().mean().backward()
    assert all(torch.equal(read.forget.grad, torch.zeros(4)) for read in held.state_reads.values())


def test_a_longer_reach_at_scoring_follows_the_power_law(monkeypatch):
    _trainer(monkeypatch)
    import eval_stream

    model = _slopes(_stack(read_reach=96, read_forget=True), 0.4, -0.7, 0.1, -1.3).eval()
    trained = model.state_reads["1"].aged().detach()
    eval_stream.lengthen_reach(model, 400)
    longer = model.state_reads["1"].aged().detach()
    assert longer.shape == (4, 400) and torch.equal(longer[:, :96], trained)
    # Every doubling of the age adds the same step, however far beyond the training the cell lies.
    slopes = torch.tensor([0.4, -0.7, 0.1, -1.3])
    torch.testing.assert_close(longer[:, 94] - longer[:, 47], slopes, rtol=0, atol=1e-5)
    torch.testing.assert_close(longer[:, 376] - longer[:, 188], slopes, rtol=0, atol=1e-5)
    with torch.no_grad():
        counts, _ = model.stream(_bytes(length=400))
    assert torch.isfinite(counts.float()).all()


@pytest.mark.parametrize("settings", [dict(read_forget=True, read_window=1), dict(read_forget=True, read_beats=8),
                                      dict(read_forget_rate=-1.0), dict(read_forget_rate=math.inf),
                                      dict(read_forget_rate=True)])
def test_a_slope_needs_a_window_and_a_finite_rate(settings):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **settings)))


def test_a_fork_turns_the_slope_on_where_its_windows_grow_and_its_control_holds_it(monkeypatch, tmp_path):
    trainer, _ = _trainer(monkeypatch)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "stack"))
    import fork_state

    # The parent reads its window of 16 cells and no further, so its slopes stay at zero.
    assert _train(trainer, monkeypatch, tmp_path / "parent", "--read-forget", "--stop-at", "6") is None
    longer = ["--read-forget", "--ctx", "32", "--bs", "1", "--read-reach", "32"]
    for name, extra in (("forgets", []), ("held", ["read_forget_rate=0"])):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(sys, "argv", ["fork_state.py", str(tmp_path / "parent" / "model.pt.resume"),
                                          str(tmp_path / name / "model.pt.resume"), "--tag", name,
                                          "--set", "ctx=32", "bs=1", "read_reach=32", *extra])
        fork_state.main()
    forgets = _train(trainer, monkeypatch, tmp_path / "forgets", *longer, "--tag", "forgets")
    held = _train(trainer, monkeypatch, tmp_path / "held", *longer, "--read-forget-rate", "0", "--tag", "held")
    assert forgets["resumed_at_steps"] == held["resumed_at_steps"] == [6]
    assert forgets["cfg"]["read_forget"] and held["cfg"]["read_forget_rate"] == 0
    slopes = {}
    for name in ("forgets", "held"):
        model, _ = trainer.load_checkpoint(tmp_path / name / "model.pt", torch.device("cpu"))
        slopes[name] = torch.cat([read.forget.detach() for read in model.state_reads.values()])
    # The arm that forgets has learned slopes in its six updates on longer windows. Its control has none.
    assert slopes["forgets"].abs().sum() > 0 and torch.equal(slopes["held"], torch.zeros(4))
