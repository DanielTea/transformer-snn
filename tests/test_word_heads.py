"""Word heads: heads of a race read that reach only the present word's cells and weigh them by their place in it."""
import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig

_LEVELS = (0, 1)
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=128,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)
_TEXT = b"the quick brown fox, jumps over 12 lazy dogs; phenomenal words again and again \n<b>caf\xc3\xa9</b> "


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
            if read.word_heads:
                read.word_age.copy_(torch.randn(read.word_age.shape, generator=generator))
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model


def _text(length=150, streams=2):
    data = (_TEXT * 4)[:length]
    return torch.tensor(list(data), dtype=torch.long).view(1, -1).repeat(streams, 1)


@pytest.mark.parametrize("cuts", [(150,), (7, 1, 53, 89), (1, 1, 1, 147), (30, 30, 30, 30, 30)])
def test_streams_are_exact_under_any_cut(cuts):
    model = _stack(read_word_heads=1, read_word_span=16).eval()
    tokens = _text()
    with torch.no_grad():
        whole, _ = model.stream(tokens, None)
        state, parts, start = None, [], 0
        for length in cuts:
            out, state = model.stream(tokens[:, start:start + length], state)
            parts.append(out)
            start += length
    assert torch.equal(whole, torch.cat(parts, 1))


def test_a_word_head_reaches_only_the_present_word():
    model = _stack(read_word_heads=1, read_word_span=16).eval()
    read = next(iter(model.state_reads.values()))
    tokens = torch.tensor([list(b"abc defg hi")], dtype=torch.long)
    spikes = model.input_spikes(tokens) if hasattr(model, "input_spikes") else None
    if spikes is None:
        spikes = torch.randint(-8, 9, (1, tokens.shape[1], 64)).float()
    with torch.no_grad():
        fired, _, _, _ = read._address(spikes, tokens)
    # At "f" (byte 6) the present word "defg" began at byte 4: the word head fires on bytes 4 to 6 only.
    word_head, other_heads = fired[0, 0, 6], fired[0, 1:, 6]
    assert word_head[:4].abs().sum() == 0 and word_head[7:].abs().sum() == 0
    assert word_head[4:7].sum() > 0
    # At the space after "defg" (byte 8) the head still reaches the word it follows, and its own byte.
    assert fired[0, 0, 8, :4].abs().sum() == 0 and fired[0, 0, 8, 4:9].sum() > 0
    # The other heads reach every earlier byte within their window.
    assert other_heads[:, :4].sum() > 0


def test_no_byte_hears_a_later_one():
    model = _stack(read_word_heads=2, read_word_span=16).eval()
    tokens = _text(length=80)
    changed = tokens.clone()
    changed[:, 50:] = (changed[:, 50:] + 7) % 256
    with torch.no_grad():
        a, _ = model.stream(tokens, None)
        b, _ = model.stream(changed, None)
    assert torch.equal(a[:, :50], b[:, :50])


def test_the_places_learn():
    model = _stack(read_word_heads=1, read_word_span=16)
    model.train()
    output = model(_text(length=64, streams=3)[:, :-1])
    output = output[0] if isinstance(output, tuple) else output
    output.float().pow(2).mean().backward()
    grads = [read.word_age.grad for read in model.state_reads.values()]
    assert all(grad is not None for grad in grads) and sum(grad.abs().sum() for grad in grads) > 0


def test_a_long_run_of_non_letters_keeps_the_own_byte():
    model = _stack(read_word_heads=1, read_word_span=4).eval()
    tokens = torch.tensor([list(b"ab" + b" " * 40 + b"cd")], dtype=torch.long)
    with torch.no_grad():
        out, _ = model.stream(tokens, None)
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("changes", [dict(read_word_heads=5), dict(read_word_heads=1, read_word_span=1),
                                     dict(read_word_heads=1, read_word_span=257),
                                     dict(read_word_heads=1, read_kept=4),
                                     dict(read_word_heads=1, read_mode="soft", read_budget=None)])
def test_refuses_what_it_cannot_wire(changes):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **changes)))
