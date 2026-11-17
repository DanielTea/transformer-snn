import numpy as np
import pytest
import torch

from snnformer import coasting_heads
from snnformer.coasting_heads import CoastingHeads, LEVELS, SEED, charge_level, coast, coast_reference


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
