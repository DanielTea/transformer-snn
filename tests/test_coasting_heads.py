from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from snnformer import coasting_heads
from snnformer.coasting_heads import CoastingHeads, LEVELS, SEED, charge_level, coast, coast_reference
from snnformer.event_lm import EventLM, EventLMConfig

_BASE = EventLMConfig(dim=32, hidden=16, depth=2, route="steered_correction_code", carry=False,
                      compact_delays=True, compact_axon_state=True, compact_trace_state=True, local_trace=True,
                      activity_inhibition=True, resolution=4, trace_arrival="selected", trace_arrival_groups=2)
_HEADS = dict(head_orders=(1, 2, 4), head_cells=64, head_gate=1, head_tape=32)
_SYNAPSES = dict(synapse_orders=(1, 2, 4), synapse_groups=3, synapse_cells=16, synapse_levels=2, synapse_gate=1)


def _compiled():
    try:
        coasting_heads.library()
    except (RuntimeError, OSError) as error:
        pytest.skip(f"no compiled replay: {error}")


def _stream(seed, length=3000):
    rng = np.random.default_rng(seed)
    phrases = [rng.integers(97, 101, size=rng.integers(3, 40)).astype(np.uint8) for _ in range(12)]
    data = np.concatenate([phrases[i] for i in rng.integers(0, len(phrases), size=length // 10)])[:length]
    surprise = rng.integers(0, 3, size=len(data)).astype(np.int16)
    return data, surprise


@pytest.mark.parametrize("setting", [
    dict(order=3, cells=64, gate=1, tape=0, lookup=0, miss=0, relock=0),
    dict(order=2, cells=16, gate=0, tape=128, lookup=0, miss=0, relock=0),
    dict(order=6, cells=256, gate=1, tape=512, lookup=1, miss=2, relock=0),
    dict(order=4, cells=64, gate=2, tape=64, lookup=0, miss=1, relock=4),
    dict(order=3, cells=64, gate=1, tape=0, lookup=0, miss=0, relock=0, lifetime=1),
    dict(order=2, cells=64, gate=1, tape=256, lookup=0, miss=0, relock=0, lifetime=5),
])
def test_compiled_replay_matches_the_reference(setting):
    _compiled()
    data, surprise = _stream(3)
    pred, charge, _ = coast(data, surprise, **setting)
    expected_pred, expected_charge = coast_reference(data, surprise, seed=(SEED + setting["order"]) & 0xFFFFFFFF, **setting)
    assert np.array_equal(pred, expected_pred)
    assert np.array_equal(charge, expected_charge)


def test_a_head_follows_a_repeated_passage_and_dies_on_the_first_wrong_byte():
    _compiled()
    # Every byte of the passage is distinct, so each context addresses one cell.
    passage = np.arange(40, 85, dtype=np.uint8)
    data = np.concatenate((passage, passage[:30], [35], passage[31:], [0])).astype(np.uint8)
    surprise = np.ones(len(data), dtype=np.int16)
    pred, charge, _ = coast(data, surprise, order=3, cells=1024, gate=1, tape=0, lookup=0, miss=0, relock=0)
    start = len(passage)
    # Three bytes of the repeat address the tape. From then on the head predicts every byte of the passage.
    follows = np.arange(start + 2, start + 29)
    assert np.array_equal(pred[follows], data[follows + 1])
    assert np.array_equal(charge[follows], np.arange(len(follows)))
    # It predicted the passage's byte where '#' arrived, and that miss removes its charge.
    assert pred[start + 29] == passage[30] and charge[start + 29] == 27
    assert charge[start + 30] == 0


def test_the_prediction_is_causal():
    _compiled()
    data, surprise = _stream(5)
    pred, charge, _ = coast(data, surprise, order=3, cells=256, gate=1, tape=256, lookup=0, miss=0, relock=0)
    changed = data.copy()
    changed[2000:] = 255 - changed[2000:]
    other_pred, other_charge, _ = coast(changed, surprise, order=3, cells=256, gate=1, tape=256, lookup=0, miss=0, relock=0)
    assert np.array_equal(pred[:1999], other_pred[:1999])
    assert np.array_equal(charge[:1999], other_charge[:1999])


def test_recall_spikes_follow_the_fitted_strength_of_the_charge_level():
    _compiled()
    data, surprise = _stream(7, length=600)
    heads = CoastingHeads((2, 4), 256, 1, 0)
    heads.readout[:, 1:] = torch.arange(1, 2 * LEVELS + 1, dtype=torch.int32).view(2, LEVELS)
    tokens = torch.from_numpy(data[None].astype(np.int64))
    spikes = heads(tokens, torch.from_numpy(surprise[None].astype(np.int64)))
    assert spikes.shape == (1, len(data), 256)
    expected = np.zeros((len(data), 256), dtype=np.int64)
    padded = np.concatenate((data, np.zeros(1, dtype=np.uint8)))
    for head, order in enumerate((2, 4)):
        pred, charge, _ = coast(padded, surprise, order, 256, 1, 0, 0, 0, 0)
        live = np.nonzero(pred >= 0)[0]
        np.add.at(expected, (live, pred[live]), heads.readout[head].numpy()[charge_level(charge[live])])
    assert np.array_equal(spikes[0].numpy(), expected)
    heads.muted = True
    assert int(heads(tokens, torch.from_numpy(surprise[None].astype(np.int64))).abs().sum()) == 0


def _heads(**setting):
    heads = CoastingHeads(**{**dict(orders=(1, 2, 3, 5, 8), cells=64, gate=1, tape=96), **setting})
    generator = torch.Generator().manual_seed(11)
    heads.readout[:, 1:] = torch.randint(-20, 90, (len(heads.orders), LEVELS), generator=generator, dtype=torch.int32)
    return heads


def _streams(seed, batch=2, length=1500):
    pairs = [_stream(seed + row, length) for row in range(batch)]
    return (torch.from_numpy(np.stack([data for data, _ in pairs])),
            torch.from_numpy(np.stack([surprise for _, surprise in pairs])))


@pytest.mark.parametrize("setting", [
    dict(),
    dict(tape=8, cells=16),
    dict(tape=64, miss=1),
    dict(tape=128, relock=4),
    dict(tape=256, gate=2, lookup=1),
    dict(tape=1 << 16, cells=4096),
])
@pytest.mark.parametrize("sizes", [(1500,), (1, 7, 100, 1392), (97,) * 15 + (45,), (1,) * 200 + (1300,)])
def test_carried_heads_match_the_replay_of_the_whole_stream(setting, sizes):
    _compiled()
    heads = _heads(**setting)
    tokens, surprise = _streams(13)
    # The stream is many times longer than the short tapes, so every ring cell is overwritten.
    expected = heads(tokens, surprise)
    assert expected.any()
    state, chunks = heads.new_state(2), []
    for part, errors in zip(tokens.split(sizes, dim=1), surprise.split(sizes, dim=1), strict=True):
        chunks.append(heads(part, errors, state))
    assert torch.equal(torch.cat(chunks, dim=1), expected)


def test_stepped_heads_match_the_replay_and_keep_a_bounded_state():
    _compiled()
    heads = _heads(tape=64)
    tokens, surprise = _streams(17, length=700)
    expected = heads(tokens, surprise)
    state, steps = heads.new_state(2), []
    for byte, errors in zip(tokens.unbind(1), surprise.unbind(1), strict=True):
        spikes, state = heads.step(byte, errors, state)
        steps.append(spikes)
    assert torch.equal(torch.stack(steps, dim=1), expected)
    carried = heads.new_state(2)
    heads(tokens, surprise, carried)
    for name in ("tape", "synapse", "written", "pointer", "charge"):
        assert torch.equal(getattr(state, name), getattr(carried, name))
    assert (state.cell, state.seen) == (700 % 64, 8) == (carried.cell, carried.seen)
    stored = sum(getattr(state, name)[0].numel() * getattr(state, name).element_size()
                 for name in ("tape", "synapse", "written", "pointer", "charge"))
    assert stored == heads.state_bytes == 64 + 5 * (2 * 64 + 8 + 8)


def test_muted_heads_still_follow_the_stream():
    _compiled()
    heads = _heads()
    tokens, surprise = _streams(19, length=400)
    expected = heads(tokens, surprise)
    state = heads.new_state(2)
    heads.muted = True
    assert not heads(tokens[:, :250], surprise[:, :250], state).any()
    heads.muted = False
    assert torch.equal(heads(tokens[:, 250:], surprise[:, 250:], state), expected[:, 250:])


def test_carried_heads_need_a_tape_that_holds_their_contexts():
    with pytest.raises(ValueError, match="tape"):
        CoastingHeads((2, 4), 64, 1, 0).new_state(1)
    with pytest.raises(ValueError, match="tape"):
        CoastingHeads((2, 9), 64, 1, 8).new_state(1)
    with pytest.raises(ValueError, match="tape"):
        CoastingHeads((2, 4), 64, 1, 1 << 17).new_state(1)


def _decoder(clocked, synapses=False, seed=29):
    torch.manual_seed(seed)
    cfg = replace(_BASE, **_HEADS, **(_SYNAPSES if synapses else {}))
    if clocked:
        cfg = replace(cfg, depth=4, residual_correction_start=2, clock_start=2, clock_threshold=24)
    model = EventLM(cfg).eval()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed)
        model.read_heads.readout[:, 1:] = torch.randint(1, 40, (3, LEVELS), generator=generator, dtype=torch.int32)
        if synapses:
            model.fast_synapses.readout[:, 1:] = torch.tensor([[2, 9], [5, 17], [8, 30]], dtype=torch.int32)
    return model


@pytest.mark.parametrize("clocked", [False, True])
@pytest.mark.parametrize("synapses", [False, True])
def test_decoder_with_heads_agrees_in_float_integer_chunked_and_streaming_form(clocked, synapses):
    _compiled()
    model = _decoder(clocked, synapses)
    native = model.to_integer()
    # The stream is three times the tape, so the heads read cells that were overwritten.
    tokens = torch.randint(0, 4, (3, 97), generator=torch.Generator().manual_seed(8))
    expected = native(tokens)
    assert torch.equal(expected, model(tokens).to(torch.int32))
    native.read_heads.muted = True
    silent = native(tokens)
    native.read_heads.muted = False
    assert not torch.equal(silent, expected)
    plain = EventLM(replace(model.cfg, head_orders=None)).eval()
    plain.load_state_dict(model.state_dict(), strict=False)
    assert torch.equal(plain.to_integer()(tokens), silent)
    for sizes in ((97,), (1, 5, 40, 51), (13,) * 7 + (6,)):
        chunks, state = [], None
        for part in tokens.split(sizes, dim=1):
            counts, state = native.forward_chunk(part, state)
            chunks.append(counts)
        assert torch.equal(torch.cat(chunks, dim=1), expected)
    steps, stream = [], None
    for byte in tokens.unbind(1):
        counts, stream = native.step(byte, stream)
        steps.append(counts)
    assert torch.equal(torch.stack(steps, dim=1), expected)
    for name in ("tape", "synapse", "written", "pointer", "charge"):
        assert torch.equal(getattr(stream.heads, name), getattr(state.heads, name))
    assert stream.heads.written.any() and (stream.heads.cell, stream.heads.seen) == (97 % 32, 4)
    # The clock or the fast synapses already carry the forecast spikes. Otherwise the heads keep their own copy.
    kept = not clocked and not synapses
    assert (stream.heads.forecast is not None) == kept == (state.heads.forecast is not None)
    if kept:
        assert stream.heads.forecast.dtype == torch.int8
        assert torch.equal(stream.heads.forecast, state.heads.forecast)


def test_the_stream_reports_live_heads_and_their_recall_spikes():
    _compiled()
    native = _decoder(True).to_integer()
    tokens = torch.randint(0, 4, (2, 97), generator=torch.Generator().manual_seed(8))
    native.read_heads.capture = []
    live = recall = 0
    state = None
    for part in tokens.split((40, 57), dim=1):
        stats = {}
        _, state = native.forward_chunk(part, state, stats=stats)
        live += int(stats["live_tape_heads"])
        recall += int(stats["tape_recall_spikes"])
    captured = native.read_heads.capture
    pred = np.concatenate([part for part, _, _ in captured], axis=-1)
    charge = np.concatenate([part for _, part, _ in captured], axis=-1)
    strength = native.read_heads.readout.numpy().astype(np.int64)
    expected = sum(int(np.abs(strength[head, charge_level(charge[:, head][pred[:, head] >= 0])]).sum())
                   for head in range(3))
    assert 0 < live == int((pred >= 0).sum()) <= 3 * tokens.numel()
    assert recall == expected > 0


def test_heads_in_the_decoder_see_only_the_past():
    _compiled()
    native = _decoder(True).to_integer()
    generator = torch.Generator().manual_seed(4)
    tokens = torch.randint(0, 4, (2, 80), generator=generator)
    other = tokens.clone()
    other[:, 50:] = torch.randint(0, 4, (2, 30), generator=generator)
    assert torch.equal(native(tokens)[:, :50], native(other)[:, :50])


def test_fitted_strengths_attach_to_a_checkpoint_without_changing_a_weight(monkeypatch, tmp_path):
    import json
    import sys
    from pathlib import Path

    _compiled()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import attach_head_readout
    from train_event_lm import load_checkpoint

    plain = EventLM(replace(_BASE, depth=4, residual_correction_start=2, clock_start=2, clock_threshold=24)).eval()
    source = {"format_version": 1, "model_type": "EventLM", "task": "event_lm", "run_name": "plain",
              "cfg": asdict(plain.cfg), "model_kwargs": {}, "args": {}, "state_dict": plain.state_dict()}
    torch.save(source, tmp_path / "plain.pt")
    strengths = torch.randint(1, 40, (3, LEVELS), generator=torch.Generator().manual_seed(5)).tolist()
    fitted = {"source_run": "plain", "fit": "a test stream",
              "setting": {"cells": 64, "gate": 1, "tape": 32, "lookup": 0, "miss": 0, "relock": 0, "orders": [1, 2, 4]},
              "head_strength_spikes": strengths}
    (tmp_path / "heads.json").write_text(json.dumps(fitted))
    monkeypatch.setattr(sys, "argv", ["attach_head_readout.py", "--checkpoint", str(tmp_path / "plain.pt"),
                                      "--readout", str(tmp_path / "heads.json"),
                                      "--output", str(tmp_path / "heads.pt")])
    attach_head_readout.main()
    model, checkpoint = load_checkpoint(tmp_path / "heads.pt", torch.device("cpu"))
    assert checkpoint["run_name"] == "plain_fitted_heads" and checkpoint["read_heads"]["fit"] == "a test stream"
    assert model.read_heads.orders == (1, 2, 4) and model.read_heads.tape == 32
    assert model.read_heads.readout[:, 1:].tolist() == strengths and not model.read_heads.readout[:, 0].any()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, checkpoint["state_dict"][name])
    # The attached decoder is the plain decoder plus the heads' recall spikes, in the stream as in one pass.
    tokens = torch.randint(0, 4, (2, 97), generator=torch.Generator().manual_seed(8))
    native = model.to_integer()
    expected = native(tokens)
    native.read_heads.muted = True
    assert torch.equal(native(tokens), plain.to_integer()(tokens))
    native.read_heads.muted = False
    chunks, state = [], None
    for part in tokens.split((40, 57), dim=1):
        counts, state = native.forward_chunk(part, state)
        chunks.append(counts)
    assert torch.equal(torch.cat(chunks, dim=1), expected) and not torch.equal(expected, plain.to_integer()(tokens))
