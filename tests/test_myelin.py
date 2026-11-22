"""Myelinated fibres: every fibre of a cell layer learns its own conduction delay (granule_myelin)."""
import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig

_LEVELS = (0, 1)
# The small stack of the stream-form tests, with a fan of one stream so that every channel has one fibre.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent; give them output synapses so that the cells shape the stream.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=120, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(seed))


def _shifted(model, low=-0.15, high=0.15, seed=3):
    """Move every delay off its start by up to about ten bytes, so that the fibres hear other bytes."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in model.granule_layers.values():
            layer.myelin_shift.copy_(torch.empty(layer.myelin_shift.shape).uniform_(low, high, generator=generator))
    return model


@pytest.mark.parametrize("cuts", [(120,), (60, 60), (47, 1, 9, 63), (1, 2, 3, 114)])
def test_streams_are_exact_under_any_cut(cuts):
    model = _shifted(_stack(granule_myelin=16)).eval()
    tokens = _bytes()
    with torch.no_grad():
        whole, _ = model.stream(tokens, None)
        state, parts, start = None, [], 0
        for length in cuts:
            out, state = model.stream(tokens[:, start:start + length], state)
            parts.append(out)
            start += length
    assert torch.equal(whole, torch.cat(parts, 1))


def test_start_equals_fixed_delays_and_learned_delays_differ():
    myelinated = _stack(granule_myelin=16).eval()
    fixed = EventLM(EventLMConfig(**_STACK)).eval()
    missing, unexpected = fixed.load_state_dict(myelinated.state_dict(), strict=False)
    assert not missing and all("myelin" in name for name in unexpected)
    tokens = _bytes()
    with torch.no_grad():
        assert torch.equal(myelinated.stream(tokens, None)[0], fixed.stream(tokens, None)[0])
        _shifted(myelinated)
        assert not torch.equal(myelinated.stream(tokens, None)[0], fixed.stream(tokens, None)[0])


def test_no_byte_hears_a_later_one():
    model = _shifted(_stack(granule_myelin=16)).eval()
    tokens = _bytes(length=60)
    changed = tokens.clone()
    changed[:, 40:] = (changed[:, 40:] + 1) % 256
    with torch.no_grad():
        a, _ = model.stream(tokens, None)
        b, _ = model.stream(changed, None)
    assert torch.equal(a[:, :40], b[:, :40])
    assert not torch.equal(a[:, 40:], b[:, 40:])


def test_a_fibre_hears_its_channel_as_it_stood_its_delay_ago():
    model = _stack(granule_myelin=16)
    layer = next(iter(model.granule_layers.values()))
    spikes = torch.arange(1.0, 41.0).view(1, 40, 1) * torch.ones(1, 1, 64)
    with torch.no_grad():
        index = (0, 5)
        layer.myelin_shift.zero_()
        start = int(layer.myelin_start[index])
        layer.myelin_shift.view_as(layer.myelin_start)[index] = (9 - start) / 64
        heard = layer.myelinated(spikes)
    assert layer.delay_bytes()[index].item() == pytest.approx(9)
    # The value a fibre brings at byte t is its channel at byte t - 9, and silence before the stream began.
    assert torch.equal(heard[0, 9:, 0, 5], spikes[0, :31, 0])
    assert torch.equal(heard[0, :9, 0, 5], torch.zeros(9))


def test_delays_learn_from_the_loss():
    model = _stack(granule_myelin=16)
    model.train()
    tokens = _bytes(length=64, streams=4)
    output = model(tokens[:, :-1])
    output = output[0] if isinstance(output, tuple) else output
    output.float().pow(2).mean().backward()
    grads = [layer.myelin_shift.grad for layer in model.granule_layers.values()]
    assert all(grad is not None for grad in grads)
    assert sum(grad.abs().sum() for grad in grads) > 0


def test_the_stream_keeps_the_longest_delay_and_no_more():
    model = _stack(granule_myelin=12).eval()
    with torch.no_grad():
        _, state = model.stream(_bytes(length=40), None)
    found = []

    def visit(value):
        if hasattr(value, "late") and hasattr(value, "held"):
            found.append(value.late)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item)

    visit(state)
    assert found and all(late.shape[1] == 11 for late in found)


@pytest.mark.parametrize("changes", [dict(granule_myelin=1), dict(granule_myelin=65),
                                     dict(granule_myelin=16, granule_traces=1),
                                     dict(granule_myelin=16, granule_fan=128),
                                     dict(granule_myelin=3, granule_delays=4)])
def test_refuses_what_it_cannot_wire(changes):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **changes)))
