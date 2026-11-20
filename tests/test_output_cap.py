import math

import pytest
import torch

from snnformer.event_lm import _OUTPUT_CAP, EventLM, EventLMConfig

_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=(0, 1), read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=(0, 1), granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True)


def _stack(**settings):
    torch.manual_seed(0)
    return EventLM(EventLMConfig(**dict(_STACK, **settings)))


def _bits(counts, resolution=16):
    # A count of an output neuron is an eighth of a bit over the resolution.
    return counts / (8 * resolution)


def test_an_output_neuron_has_its_say_up_to_sixteen_bits_unless_told_otherwise():
    assert EventLMConfig.output_cap == _OUTPUT_CAP == 128
    model = _stack()
    assert model.output_emit.cap == 128 * 16 and _bits(model.output_emit.cap) == 16
    wide = _stack(output_cap=512)
    assert wide.output_emit.cap == 512 * 16 and _bits(wide.output_emit.cap) == 64
    # The cap adds no synapse and moves none.
    assert set(model.state_dict()) == set(wide.state_dict())
    assert all(torch.equal(value, wide.state_dict()[name]) for name, value in model.state_dict().items())


@pytest.mark.parametrize("cap", [32, 128, 512])
def test_the_output_fires_whole_counts_up_to_its_cap_and_no_further(cap):
    model = _stack(output_cap=cap)
    fire = model.output_emit
    charge = torch.tensor([0.0, 7.0, 8.0, -8.0, 8.0 * 40 * 16, -8.0 * 300 * 16, 1e7, -1e7])
    counts = fire(charge)
    assert torch.equal(counts, counts.round())
    assert counts[:4].tolist() == [0, 1, 1, -1]
    largest = cap * 16
    assert counts[4] == min(40 * 16, largest) and counts[5] == -min(300 * 16, largest)
    assert counts[6:].tolist() == [largest, -largest]
    # The largest score between two bytes is twice the cap: the odds of 2 ** (cap / 4) to one.
    assert math.isclose(_bits(2 * largest), cap / 4)


def test_a_higher_cap_changes_only_the_bytes_that_reached_the_old_one():
    narrow, wide = _stack(output_cap=1), _stack(output_cap=128)
    with torch.no_grad():
        for model in (narrow, wide):
            model.output.weight.normal_(0, 0.5, generator=torch.Generator().manual_seed(3))
    tokens = torch.randint(0, 256, (2, 60), generator=torch.Generator().manual_seed(2))
    first, second = narrow(tokens), wide(tokens)
    assert first.abs().max() == 16 and second.abs().max() > 16
    assert torch.equal(first, second.clamp(-16, 16))
    at_cap = first.abs() == 16
    assert 0 < at_cap.float().mean() < 1 and torch.equal(first[~at_cap], second[~at_cap])


def test_the_forecast_population_stays_within_a_byte():
    model = _stack(output_cap=4096)
    assert model.forecast_emit.cap == 127 and _stack().forecast_emit.cap == 127


@pytest.mark.parametrize("cap", [0, -8, 128.0, 1 << 18])
def test_a_model_refuses_an_output_cap_it_cannot_count(cap):
    with pytest.raises(ValueError, match="output cap"):
        _stack(output_cap=cap)
