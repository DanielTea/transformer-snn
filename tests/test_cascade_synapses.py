"""Fast output synapses that consolidate themselves: every synapse holds a cascade of charges, and what a byte
taught spreads from the stage that the output hears into slower and larger ones."""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from snnformer.event_lm import _TEACH_LIMIT, EventLM, EventLMConfig
from test_teaching_spikes import _STACK, _TAUGHT, _bytes, _pieces, _stack


def _cascade(stages, charges):
    """A state of fast synapses with these charges in its stages, as [stages, ...]."""
    charges = torch.as_tensor(charges, dtype=torch.float64)
    return SimpleNamespace(fast=charges[0].clone(), deep=charges[1:].clone())


def _consolidated(state, positions):
    for position in positions:
        EventLM._consolidate(state, position)
    return state


def _lesson(stages, ages, size=1 << 30):
    """The share of one lesson that the first stage still holds at each of these ages, in bytes."""
    state, shares, age = _cascade(stages, [size] + [0] * (stages - 1)), [], 0
    for wanted in ages:
        _consolidated(state, range(age + 1, wanted + 1))
        age = wanted
        shares.append(float(state.fast) / size)
    return shares


def test_one_stage_is_the_rule_that_does_not_consolidate():
    plain, single, tokens = _stack(**_TAUGHT), _stack(**dict(_TAUGHT, teach_cascade=1)), _bytes(length=200)
    fired, state = _pieces(single, tokens, (200,))
    assert torch.equal(fired, _pieces(plain, tokens, (200,))[0]) and state.taught.deep is None


@pytest.mark.parametrize("stages", [2, 4, 6])
@pytest.mark.parametrize("cuts", [(1,) * 300, (7,) * 42 + (6,), (47, 1, 49, 203), (128, 172), (299, 1)])
def test_a_consolidating_stream_cut_anywhere_fires_the_spikes_of_one_pass(stages, cuts):
    model, tokens = _stack(**dict(_TAUGHT, teach_cascade=stages)), _bytes(length=300)
    whole, after = _pieces(model, tokens, (300,))
    fired, state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole)
    assert torch.equal(state.taught.fast, after.taught.fast) and torch.equal(state.taught.deep, after.taught.deep)
    assert after.taught.deep.shape == (stages - 1, 2, 256, 96)


def test_consolidation_changes_what_the_output_hears():
    plain, model, tokens = _stack(**_TAUGHT), _stack(**dict(_TAUGHT, teach_cascade=4)), _bytes(length=300)
    once, _ = _pieces(plain, tokens, (300,))
    fired, state = _pieces(model, tokens, (300,))
    # Until the eighth byte has taught, no charge has moved. Then the first stage gives charge to the slower ones.
    assert torch.equal(fired[:, :8], once[:, :8]) and not torch.equal(fired, once)
    assert state.taught.deep.abs().sum() > 0


def test_the_stages_hold_whole_numbers_that_add_up_to_what_was_taught():
    generator = torch.Generator().manual_seed(4)
    stages = 6
    # Charges at levels up to the bound of a synapse, stage k holding 2 ** k times its level.
    levels = torch.randint(-_TEACH_LIMIT, _TEACH_LIMIT, (stages, 3, 50), generator=generator, dtype=torch.int64)
    charges = levels.double() * (2.0 ** torch.arange(stages, dtype=torch.float64))[:, None, None]
    state = _cascade(stages, charges)
    total = charges.sum(0)
    low, high = levels.double().amin(0), levels.double().amax(0)
    for position in range(1, 4097):
        _consolidated(state, (position,))
        if position % 512 == 0:
            held = torch.cat([state.fast[None], state.deep])
            assert torch.equal(held, held.round()) and torch.equal(held.sum(0), total)
            reached = held / (2.0 ** torch.arange(stages, dtype=torch.float64))[:, None, None]
            # No flow overshoots: every level stays within the levels that the synapse started from.
            assert (reached >= low - 1).all() and (reached <= high + 1).all()


def test_charge_of_either_sign_consolidates_alike():
    generator = torch.Generator().manual_seed(6)
    charges = torch.randint(-(1 << 40), 1 << 40, (5, 200), generator=generator).double()
    positive = _consolidated(_cascade(5, charges), range(1, 2049))
    negative = _consolidated(_cascade(5, -charges), range(1, 2049))
    assert torch.equal(positive.fast, -negative.fast) and torch.equal(positive.deep, -negative.deep)


@pytest.mark.parametrize("stages", [2, 3, 4])
def test_a_lesson_settles_at_its_share_of_all_the_stages(stages):
    # Once a lesson has spread through every stage, the first holds 1 / (2 ** stages - 1) of it: 1/3, 1/7, 1/15.
    share, = _lesson(stages, (1 << 15,))
    assert share == pytest.approx(1 / (2 ** stages - 1), abs=1e-6)


def test_a_lesson_fades_as_a_power_of_its_age_and_never_wholly():
    ages = (1, 7, 8, 64, 1024, 65536, 1 << 20)
    shares = dict(zip(ages, _lesson(6, ages)))
    # The first seven bytes after a lesson hear all of it, the eighth half.
    assert shares[1] == shares[7] == 1 and shares[8] == 0.5
    # From a word to a book it fades as about the cube root of its age.
    slope = math.log(shares[65536] / shares[64]) / math.log(65536 / 64)
    assert -0.4 < slope < -0.3
    # It never fades below the share of the last stage, which keeps it.
    assert shares[1 << 20] == pytest.approx(1 / 63, abs=1e-6)
    held = list(shares.values())
    assert all(earlier >= later for earlier, later in zip(held, held[1:]))


def test_a_consolidating_state_can_be_continued_more_than_once():
    model, tokens = _stack(**dict(_TAUGHT, teach_cascade=3)), _bytes(length=200)
    _, state = _pieces(model, tokens[:, :100], (100,))
    held = state.taught.deep.clone()
    first, _ = _pieces(model, tokens[:, 100:], (100,), state)
    again, _ = _pieces(model, tokens[:, 100:], (37, 63), state)
    assert torch.equal(state.taught.deep, held) and torch.equal(first, again)


def test_the_stages_are_given_with_the_fast_synapses_after_training():
    model, built, tokens = _stack(), _stack(**dict(_TAUGHT, teach_cascade=5)), _bytes(length=200)
    model.teach((1,), 1 << 24, cascade=5)
    assert model.cfg == built.cfg
    assert torch.equal(_pieces(model, tokens, (200,))[0], _pieces(built, tokens, (200,))[0])
    model.teach(None)
    assert model.cfg.teach_cascade == 1


@pytest.mark.parametrize("stages", [0, 7, 2.0, True, -1])
def test_a_cascade_has_one_to_six_stages(stages):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **dict(_TAUGHT, teach_cascade=stages))))
    with pytest.raises(ValueError):
        _stack().teach((1,), 1 << 24, cascade=stages)


def test_stages_belong_to_fast_synapses():
    with pytest.raises(ValueError, match="stages"):
        EventLM(EventLMConfig(**dict(_STACK, teach_cascade=4)))


def test_the_scorer_consolidates_every_stream_by_itself(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import eval_stream

    data = torch.from_numpy(np.random.default_rng(1).integers(0, 256, 1000, dtype=np.uint8))
    model = _stack()
    streams = eval_stream.stream_rows(data, 4, 30)
    model.teach((1,), 1 << 24)
    once, _ = eval_stream.score(model, streams, 50, torch.device("cpu"))
    model.teach((1,), 1 << 24, cascade=4)
    taught, _ = eval_stream.score(model, streams, 50, torch.device("cpu"))
    other, _ = eval_stream.score(model, streams, 273, torch.device("cpu"))
    assert not np.array_equal(taught, once) and taught.sum() == pytest.approx(other.sum(), rel=1e-12)
