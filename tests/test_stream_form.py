"""A stack takes a stream in pieces of any length and fires the spikes of one pass over the whole of it."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from snnformer.event_lm import (_GOLDEN, EventLM, EventLMConfig, FibreState, ReadTape, StreamState, _late,
                                _spread_phases)

_LEVELS = (0, 1)
# Reads that reach 48 cells back and cells after both levels of a small stack.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True)
# Late fibres at the reads, delayed fibres at the cells, and pools that set the size of a spike.
_LATE = dict(read_delays=4)
_DELAYED = dict(granule_fan=128, granule_delays=4)
_POOLED = dict(read_cap=128, granule_cap=128, pool_fine=16, pool_rest=4, pool_rate=32, output_pool=128)
_DESIGNS = [dict(), _LATE, dict(_LATE, read_delay_probes=True), _DELAYED, _POOLED, dict(_POOLED, pool_lifts_ceiling=True),
            dict(_LATE, **_DELAYED, **_POOLED)]


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=160, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(seed))


def _pieces(model, tokens, cuts):
    """The counts of a stream that arrives in pieces of the lengths in ``cuts``, and the state after it."""
    counts, state, start = [], None, 0
    for length in cuts:
        fired, state = model.stream(tokens[:, start:start + length], state)
        counts.append(fired)
        start += length
    assert start == tokens.shape[1]
    return torch.cat(counts, 1), state


@pytest.mark.parametrize("design", _DESIGNS)
@pytest.mark.parametrize("cuts", [(160,), (80, 80), (48, 48, 64), (47, 1, 49, 63), (7, 13, 100, 40), (1, 2, 3, 154),
                                  (130, 30)])
def test_a_stream_cut_anywhere_fires_the_spikes_of_one_pass(design, cuts):
    model, tokens = _stack(**design), _bytes()
    whole = model(tokens)
    fired, state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole) and torch.equal(whole, whole.round())
    assert state.position == 160


@pytest.mark.parametrize("design", [dict(), dict(_LATE, **_DELAYED, **_POOLED)])
def test_a_stream_arrives_a_byte_at_a_time(design):
    model, tokens = _stack(**design).eval(), _bytes(length=70)
    with torch.no_grad():
        fired, _ = _pieces(model, tokens, (1,) * 70)
        assert torch.equal(fired, model(tokens))


def test_the_state_holds_what_the_next_bytes_can_reach():
    model, tokens = _stack(**_LATE, **_DELAYED), _bytes()
    _, state = model.stream(tokens[:, :20])
    assert state.position == 20 and set(state.reads) == set(state.cells) == {"0", "1"}
    for tape in state.reads.values():
        # Every cell so far, as [streams, heads, cells, channels], and the last three bytes for the late fibres.
        assert tape.tag.shape == tape.payload.shape == (2, 4, 20, 8) and tape.late.shape == (2, 3, 64)
    _, state = model.stream(tokens[:, 20:], state)
    for tape in state.reads.values():
        # A byte reaches the cells of the last 48 bytes with its own, so 47 stay on the tape.
        assert tape.tag.shape == tape.payload.shape == (2, 4, 47, 8) and tape.late.shape == (2, 3, 64)
        assert not (tape.tag.requires_grad or tape.payload.requires_grad or tape.late.requires_grad)
    for fibres in state.cells.values():
        assert fibres.late.shape == (2, 3, 64) and fibres.held is None and not fibres.late.requires_grad
    # A stack without late or delayed fibres keeps the tapes alone.
    _, state = _stack().stream(tokens[:, :20])
    assert all(tape.late is None for tape in state.reads.values())
    assert all(fibres.late is None and fibres.held is None for fibres in state.cells.values())


def test_a_state_serves_any_number_of_continuations():
    model, tokens = _stack(**_LATE, **_DELAYED), _bytes()
    _, state = model.stream(tokens[:, :90])
    kept = {key: (tape.tag.clone(), tape.payload.clone(), tape.late.clone()) for key, tape in state.reads.items()}
    first, after = model.stream(tokens[:, 90:], state)
    second, _ = model.stream(tokens[:, 90:], state)
    assert torch.equal(first, second) and state.position == 90 and after.position == 160
    for key, tape in state.reads.items():
        assert all(torch.equal(now, before) for now, before in zip((tape.tag, tape.payload, tape.late), kept[key]))
    # Another continuation of the same stream fires other spikes from its first byte on.
    other, _ = model.stream(_bytes(seed=9)[:, 90:], state)
    assert not torch.equal(other[:, 0], first[:, 0]) or not torch.equal(other, first)


def test_the_bytes_before_a_piece_change_what_it_fires():
    model, tokens = _stack(), _bytes()
    cold, _ = model.stream(tokens[:, 100:])
    _, state = model.stream(tokens[:, :100])
    warm, _ = model.stream(tokens[:, 100:], state)
    assert torch.equal(warm, model(tokens)[:, 100:]) and not torch.equal(warm, cold)
    # A read reaches 48 cells back, and each level reads the level below: bytes far behind still matter.
    changed = tokens.clone()
    changed[:, :40] = _bytes(seed=5)[:, :40]
    assert not torch.equal(model(changed)[:, 100:], model(tokens)[:, 100:])


def test_a_stream_sees_no_later_byte():
    model, tokens = _stack(**_LATE, **_DELAYED), _bytes()
    changed = tokens.clone()
    changed[:, 120:] = _bytes(seed=7)[:, 120:]
    first, _ = _pieces(model, tokens, (100, 60))
    second, _ = _pieces(model, changed, (100, 60))
    assert torch.equal(first[:, :120], second[:, :120]) and not torch.equal(first[:, 120:], second[:, 120:])


def test_a_piece_learns_from_what_it_fires_itself():
    model, tokens = _stack(**_LATE, **_DELAYED, **_POOLED), _bytes()
    _, state = model.stream(tokens[:, :100])
    model.zero_grad()
    fired, _ = model.stream(tokens[:, 100:], state)
    fired.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        if name.startswith(("state_reads", "granule_layers")) and parameter.ndim >= 2:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    # The ages beyond the piece's own 60 bytes address cells of the tape, and they learn too.
    for read in model.state_reads.values():
        assert (read.age.grad[:, 60:48 + 60].abs().sum(0) > 0).any() if read.age.shape[1] > 60 else True
        assert read.age.grad[:, :48].abs().sum() > 0


def test_the_pulses_of_the_pieces_are_those_of_one_pass():
    model, tokens = _stack(**_LATE, **_DELAYED), _bytes()
    whole = []
    model(tokens, memory_pulses=whole)
    pieces, state = [], None
    for start in range(0, 160, 40):
        _, state = model.stream(tokens[:, start:start + 40], state, memory_pulses=pieces)
    assert torch.stack(pieces).sum() == torch.stack(whole).sum() > 0


def test_slow_synapses_keep_their_charge_through_a_stream():
    # Fibres from one stream: its channels now, a byte late, and through two kinds of slow synapse.
    model, tokens = _stack(granule_delays=2, granule_traces=2), _bytes()
    spikes = model.input_emit(model.input(tokens) * 16)
    layer = model.granule_layers["0"]
    whole = layer.held(spikes)
    carried, parts = FibreState(), []
    for start in range(0, 160, 40):
        parts.append(layer.held(spikes[:, start:start + 40], carried))
    for kind, passed in enumerate(whole):
        joined = torch.cat([part[kind] for part in parts], 1)
        assert torch.allclose(joined, passed, rtol=1e-6, atol=1e-6)
    assert len(carried.held) == 2 and all(charge.shape == (2, 1, 64) for charge in carried.held)
    # The whole stack follows.
    fired, state = _pieces(model, tokens, (50, 50, 60))
    assert torch.allclose(fired, model(tokens), atol=1.0) and (fired != model(tokens)).float().mean() < 0.01
    assert all(fibres.held is not None for fibres in state.cells.values())


def test_a_late_fibre_brings_what_stood_before_the_piece():
    present = torch.arange(1, 7).float().reshape(1, 6, 1)
    before = torch.tensor([[[-3.0], [-2.0], [-1.0]]])
    assert _late(present, None, 2).flatten().tolist() == [0, 0, 1, 2, 3, 4]
    assert _late(present, before, 2).flatten().tolist() == [-2, -1, 1, 2, 3, 4]
    assert _late(present, before[:, 2:], 2).flatten().tolist() == [0, -1, 1, 2, 3, 4]
    # A delay longer than the piece and its past together brings silence.
    assert _late(present[:, :1], None, 3).flatten().tolist() == [0]
    assert _late(present[:, :2], before[:, 2:], 3).flatten().tolist() == [0, 0]
    assert _late(present[:, :2], before, 3).flatten().tolist() == [-3, -2]


def test_the_phases_of_the_race_are_those_of_a_cell_wherever_a_stream_stands():
    near = _spread_phases(0, 4096, torch.device("cpu"), torch.float32)
    cells = torch.arange(1, 4097, dtype=torch.float64)
    assert torch.allclose(near.double(), (cells * _GOLDEN) % 1, atol=2e-6) and 0 <= near.min() and near.max() < 1
    assert torch.equal(_spread_phases(1000, 96, torch.device("cpu"), torch.float32), near[1000:1096])
    # A billion bytes into a stream the cells still have evenly spread phases of their own.
    far = _spread_phases(10 ** 9, 4096, torch.device("cpu"), torch.float32)
    assert far.unique().numel() == 4096
    assert torch.allclose(far.sort().values, (torch.arange(4096) + 0.5) / 4096, atol=1.5 / 4096)
    # From cell to cell a phase moves on by the golden ratio, there as here.
    for phases in (near, far):
        assert torch.allclose((phases[1:] - phases[:-1]) % 1, torch.full((4095,), _GOLDEN), atol=1e-6)


def test_a_stream_far_from_its_start_fires_as_one_near_it():
    # The spikes of a race depend on the phases of its cells, and those on where the stream stands.
    model, tokens = _stack(), _bytes()
    _, state = model.stream(tokens[:, :100])
    near, _ = model.stream(tokens[:, 100:], state)
    state.position = 10 ** 9
    far, _ = model.stream(tokens[:, 100:], state)
    assert torch.equal(far, far.round()) and near.shape == far.shape
    # Other phases move single spikes of the cells, and the counts stay close.
    assert (far - near).abs().float().mean() < 0.2 * near.abs().float().mean()


@pytest.mark.parametrize("setting", [dict(read_beats=2), dict(read_phases=4), dict(read_turn=True)])
def test_the_clocks_of_a_read_are_not_wired_to_a_stream(setting):
    try:
        model = _stack(**setting)
    except ValueError:
        pytest.skip("this stack does not take the setting")
    with pytest.raises(ValueError, match="not wired to a stream"):
        model.stream(_bytes()[:, :20])


def test_only_a_stack_carries_its_state_this_way():
    blocks = EventLM(EventLMConfig(**dict(_STACK, stack=False, hidden=8, depth=2, read_after=(0,), granule_after=None)))
    with pytest.raises(ValueError, match="a stack of reads and cells"):
        blocks.stream(_bytes()[:, :20])
    model, tokens = _stack(), _bytes()
    _, state = model.stream(tokens[:, :20])
    with pytest.raises(ValueError, match="another number of streams"):
        model.stream(_bytes(streams=3)[:, 20:40], state)
    with pytest.raises(ValueError, match="a stack of reads and cells"):
        model(tokens, return_prediction=True, stream=StreamState())
    assert isinstance(state.reads["0"], ReadTape) and isinstance(state.cells["0"], FibreState)


# The trainer and the scorer.

_TRAINER = """--variant event_steered --no-carry --stack --dim 32 --hidden 4 --depth 2 --spike-resolution 16
    --router-charge-bits 8 --read-after 0,1 --read-width 16 --read-heads 2 --read-window 16 --read-mode race
    --read-budget 16 --read-resolution 4 --read-cap 128 --read-delays 4 --granule-after 0,1 --granule-clusters 1
    --granule-fan 64 --granule-cells 16 --granule-delays 4 --granule-slope --granule-cap 128 --pool-fine 16
    --pool-rest 4 --pool-rate 32 --output-pool 128 --pulse-budget 4000 --weight-average 0.9
    --ctx 16 --bs 2 --steps 12 --warmup 2 --lr 0.003 --seed 3 --eval-every 6 --eval-bs 2 --eval-batches 2
    --final-eval-batches 2 --validation-only --save-every 6 --resume""".split()


def _trainer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import train_event_lm

    data = np.random.default_rng(0).integers(0, 256, 30000, dtype=np.uint8)
    monkeypatch.setattr(train_event_lm, "load_splits", lambda path: (data[:20000], data[20000:25000], data[25000:]))
    return train_event_lm, data


def _train(trainer, monkeypatch, folder, *flags):
    monkeypatch.setattr(sys, "argv", ["train_event_lm.py", *_TRAINER, *flags, "--checkpoint", str(folder / "model.pt"),
                                      "--results-dir", str(folder / "results")])
    trainer.main()
    results = list((folder / "results").glob("*.json"))
    return json.loads(results[0].read_text()) if results else None


def test_the_trainer_trains_a_stack_on_streams(monkeypatch, tmp_path):
    trainer, data = _trainer(monkeypatch)
    drawn = []
    draw = trainer.random_batch
    monkeypatch.setattr(trainer, "random_batch",
                        lambda source, rows, length, generator: drawn.append(length) or draw(source, rows, length,
                                                                                             generator))
    record = _train(trainer, monkeypatch, tmp_path / "streams", "--stream", "3")
    # Four stretches of three windows of 16 bytes make the twelve updates.
    assert drawn == [48] * 4 and record["args"]["stream"] == 3
    for row in record["curve"]:
        assert {"val_bpb", "averaged_val_bpb", "stream_val_bpb", "averaged_stream_val_bpb"} <= set(row)
    assert record["stream_val"]["target_bytes"] == record["val"]["target_bytes"]
    assert record["averaged_stream_val"]["bpb"] > 0 and "stream_test" not in record
    # A stack that trains on windows is scored on streams as well.
    drawn.clear()
    windows = _train(trainer, monkeypatch, tmp_path / "windows")
    assert drawn == [16] * 12 and windows["args"]["stream"] is None
    assert all("stream_val_bpb" in row for row in windows["curve"])
    assert windows["curve"][-1]["val_bpb"] != record["curve"][-1]["val_bpb"]
    # A run turns from windows to streams at an update of its schedule, and is as the run on windows until then.
    drawn.clear()
    turned = _train(trainer, monkeypatch, tmp_path / "turned", "--stream", "3", "--stream-from", "6")
    assert drawn == [16] * 6 + [48] * 2
    assert turned["curve"][0] == windows["curve"][0] and turned["curve"][1]["val_bpb"] != windows["curve"][1]["val_bpb"]


def test_a_run_on_streams_that_stops_continues_as_if_it_had_not(monkeypatch, tmp_path):
    trainer, _ = _trainer(monkeypatch)
    whole = _train(trainer, monkeypatch, tmp_path / "whole", "--stream", "3")
    assert _train(trainer, monkeypatch, tmp_path / "parts", "--stream", "3", "--stop-at", "6") is None
    parts = _train(trainer, monkeypatch, tmp_path / "parts", "--stream", "3")
    assert parts["resumed_at_steps"] == [6]
    keys = ("train_bpb", "val_bpb", "averaged_val_bpb", "stream_val_bpb", "averaged_stream_val_bpb")
    assert [[row[key] for key in keys] for row in parts["curve"]] == [[row[key] for key in keys]
                                                                       for row in whole["curve"]]
    # A run on windows does not continue on streams.
    assert _train(trainer, monkeypatch, tmp_path / "other", "--stop-at", "6") is None
    with pytest.raises(SystemExit):
        _train(trainer, monkeypatch, tmp_path / "other", "--stream", "3")
    # A run that turns to streams continues through the turn as if it had not stopped.
    whole = _train(trainer, monkeypatch, tmp_path / "turn", "--stream", "3", "--stream-from", "6")
    assert _train(trainer, monkeypatch, tmp_path / "turned", "--stream", "3", "--stream-from", "6",
                  "--stop-at", "6") is None
    parts = _train(trainer, monkeypatch, tmp_path / "turned", "--stream", "3", "--stream-from", "6")
    assert [[row[key] for key in keys] for row in parts["curve"]] == [[row[key] for key in keys]
                                                                       for row in whole["curve"]]


def test_groups_of_streams_take_turns(monkeypatch, tmp_path):
    trainer, _ = _trainer(monkeypatch)
    drawn, heard = [], []
    draw = trainer.random_batch

    def record(source, rows, length, generator):
        drawn.append(length)
        stretch = draw(source, rows, length, generator)
        heard.append(stretch[0])
        return stretch

    monkeypatch.setattr(trainer, "random_batch", record)
    stream = trainer.EventLM.stream
    pieces = []
    monkeypatch.setattr(trainer.EventLM, "stream", lambda self, tokens, state=None, **options: (
        # The trainer first hands the stack two bytes to see that it takes a stream.
        pieces.append((tokens.cpu().clone(), None if state is None else state.position))
        if self.training and tokens.shape[1] == 16 else None,
        stream(self, tokens, state, **options))[1])
    record_ = _train(trainer, monkeypatch, tmp_path / "groups", "--stream", "3", "--stream-groups", "2")
    # Two groups of streams of three windows: each group draws a stretch at its first turn of every round of six.
    assert drawn == [48] * 4 and record_["args"]["stream_groups"] == 2 and len(pieces) == 12
    for turn, (tokens, position) in enumerate(pieces):
        group, piece = turn % 2, turn // 2 % 3
        stretch = heard[2 * (turn // 6) + group]
        assert torch.equal(tokens, stretch[:, piece * 16:(piece + 1) * 16])
        assert position == (None if piece == 0 else piece * 16)
    # A run with groups that stops between rounds continues as if it had not.
    whole = _train(trainer, monkeypatch, tmp_path / "whole", "--stream", "3", "--stream-groups", "2")
    assert _train(trainer, monkeypatch, tmp_path / "parts", "--stream", "3", "--stream-groups", "2",
                  "--stop-at", "6") is None
    parts = _train(trainer, monkeypatch, tmp_path / "parts", "--stream", "3", "--stream-groups", "2")
    keys = ("train_bpb", "val_bpb", "stream_val_bpb")
    assert [[row[key] for key in keys] for row in parts["curve"]] == [[row[key] for key in keys]
                                                                       for row in whole["curve"]]
    assert whole["curve"] == record_["curve"]


def test_the_trainer_hands_the_reach_to_the_model(monkeypatch, tmp_path):
    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / "reach", "--stream", "3", "--read-reach", "40")
    assert record["cfg"]["read_reach"] == 40 and record["cfg"]["read_window"] == 16
    model, _ = trainer.load_checkpoint(tmp_path / "reach" / "model.pt", torch.device("cpu"))
    assert all(read.reach == 40 and read.age.shape == (2, 16) for read in model.state_reads.values())
    with pytest.raises(SystemExit):
        _train(trainer, monkeypatch, tmp_path / "short", "--read-reach", "8")


@pytest.mark.parametrize("flags", [("--stream", "1"), ("--stream", "4"), ("--stream", "3", "--read-beats", "2"),
                                   ("--stream-from", "6"), ("--stream", "3", "--stream-from", "4"),
                                   ("--stream", "3", "--stream-from", "12"), ("--stream-groups", "2"),
                                   ("--stream", "3", "--stream-groups", "0"),
                                   ("--stream", "3", "--stream-groups", "4"),
                                   ("--stream", "2", "--stream-groups", "3", "--stream-from", "4")])
def test_the_trainer_refuses_streams_that_do_not_fit(monkeypatch, tmp_path, flags):
    trainer, _ = _trainer(monkeypatch)
    with pytest.raises(SystemExit):
        _train(trainer, monkeypatch, tmp_path / "run", *flags)


def test_the_validation_on_streams_scores_what_one_pass_scores(monkeypatch):
    trainer, data = _trainer(monkeypatch)
    model = _stack(**_LATE, **_DELAYED).eval()
    scored = trainer.evaluate_stream(model, data, context=16, batches=6, batch_size=2, device=torch.device("cpu"),
                                     event=True)
    # As many targets as twelve windows hold: three streams of two unscored and four scored pieces of 16 bytes,
    # evenly spaced.
    length, bits = 6 * 16, 0.0
    stride = (len(data) - length - 1) // 3
    with torch.no_grad():
        for stream in range(3):
            stretch = torch.from_numpy(data[stream * stride:stream * stride + length + 1].astype(np.int64))[None]
            logits = trainer.scoring_logits(model(stretch[:, :-1]), True, model.cfg.resolution)[:, 32:]
            bits += torch.nn.functional.cross_entropy(logits.flatten(0, 1), stretch[0, 33:], reduction="sum").item()
    assert scored["target_bytes"] == 3 * 4 * 16
    assert scored["bpb"] == pytest.approx(bits / (3 * 4 * 16 * math.log(2)), rel=1e-6)
    assert model.training is False


def test_a_split_is_scored_as_streams_that_run_side_by_side(monkeypatch, tmp_path):
    _trainer(monkeypatch)
    import eval_stream

    data = torch.from_numpy(np.random.default_rng(1).integers(0, 256, 1000, dtype=np.uint8))
    inputs, targets, scored = eval_stream.stream_rows(data, 4, 30)
    # 999 targets: the first stream takes 30 and a part of 243, the others hear 30 bytes and take a part each.
    assert inputs.shape == targets.shape == scored.shape == (4, 273) and scored.sum() == 999
    assert scored[0].all() and not scored[1:, :30].any() and scored[1:3, 30:].all()
    assert torch.equal(inputs[0], data[:273].long()) and torch.equal(targets[0], data[1:274].long())
    assert torch.equal(inputs[2, :60], data[486:546].long()) and torch.equal(targets[2, 30:35], data[517:522].long())
    # Every target of the split counts once.
    places = (torch.arange(4)[:, None] * 243 + torch.arange(273))[scored]
    assert torch.equal(places.sort().values, torch.arange(999))
    with pytest.raises(ValueError):
        eval_stream.stream_rows(data[:30], 4, 30)

    model = _stack(**_LATE, **_DELAYED).eval()
    bits, count = eval_stream.score(model, data, 4, 30, 50, torch.device("cpu"))
    assert count == 999 and bits.shape == (4, 6)
    # The pieces do not change the score, and one pass over each stream gives it too.
    other, _ = eval_stream.score(model, data, 4, 30, 273, torch.device("cpu"))
    assert bits.sum() == pytest.approx(other.sum(), rel=1e-9)
    with torch.no_grad():
        logits = eval_stream.scoring_logits(model(inputs), True, model.cfg.resolution)
        losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
    assert bits.sum() == pytest.approx((losses.view(4, -1) * scored).sum().item() / math.log(2), rel=1e-6)


def test_a_read_reaches_further_back_in_a_stream_than_in_training(monkeypatch):
    _trainer(monkeypatch)
    import eval_stream

    model, tokens = _stack().eval(), _bytes()
    with torch.no_grad():
        before = model(tokens)
        eval_stream.lengthen_reach(model, 48)
        assert torch.equal(model(tokens), before)
        eval_stream.lengthen_reach(model, 96)
        after, state = model.stream(tokens)
    read = model.state_reads["0"]
    # No synapse is new: the cells beyond the window take the offset of its oldest cell.
    assert read.window == 48 and read.reach == 96 and read.age.shape == (4, 48)
    assert state.reads["0"].tag.shape[2] == 95
    # The first 48 bytes have nothing beyond the window, and the later ones fire other spikes.
    assert torch.equal(after[:, :48], before[:, :48]) and not torch.equal(after[:, 48:], before[:, 48:])
    with pytest.raises(ValueError):
        eval_stream.lengthen_reach(model, 47)
    # A stack that is built with the longer reach fires the same.
    built = _stack(read_reach=96).eval()
    with torch.no_grad():
        assert torch.equal(built(tokens), after)
    assert set(built.state_dict()) == set(_stack().state_dict())
    # The reach of one level alone.
    single = _stack().eval()
    eval_stream.lengthen_reach(single, 96, levels=[1])
    assert [single.state_reads[level].reach for level in ("0", "1")] == [48, 96]
    with torch.no_grad():
        partly, state = single.stream(tokens)
    assert state.reads["0"].tag.shape[2] == 47 and state.reads["1"].tag.shape[2] == 95
    assert torch.equal(partly[:, :48], before[:, :48]) and not torch.equal(partly, after)
    with pytest.raises(ValueError):
        eval_stream.lengthen_reach(single, 96, levels=[5])


@pytest.mark.parametrize("cuts", [(160,), (80, 80), (47, 1, 49, 63), (7, 13, 100, 40), (130, 30)])
def test_a_stream_with_a_longer_reach_fires_the_spikes_of_one_pass(cuts):
    model, tokens = _stack(read_reach=96, **_LATE, **_DELAYED, **_POOLED), _bytes()
    fired, state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, model(tokens))
    assert all(tape.tag.shape[2] == 95 for tape in state.reads.values())


def test_the_oldest_age_of_the_window_learns_for_all_cells_beyond_it():
    model, tokens = _stack(read_reach=96), _bytes()
    _, state = model.stream(tokens[:, :100])
    fired, _ = model.stream(tokens[:, 100:], state)
    fired.float().square().mean().backward()
    far, near = _stack(), _bytes()
    _, state = far.stream(near[:, :100])
    fired, _ = far.stream(near[:, 100:], state)
    fired.float().square().mean().backward()
    for long, short in zip(model.state_reads.values(), far.state_reads.values()):
        assert long.age.grad[:, -1].abs().sum() > 0 and not torch.equal(long.age.grad, short.age.grad)


@pytest.mark.parametrize("setting", [dict(read_reach=47), dict(read_reach=96.0), dict(read_reach=True),
                                     dict(read_reach=96, read_beats=2), dict(read_reach=96, read_phases=4)])
def test_a_reach_is_at_least_the_window_and_takes_no_clock(setting):
    with pytest.raises(ValueError):
        _stack(**setting)
