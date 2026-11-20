"""Writing text with a stack as one stream."""

from pathlib import Path

import pytest
import torch

from snnformer.event_lm import EventLM, EventLMConfig

# Reads that reach 48 cells back and 96 cells after both levels of a small stack.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=(0, 1), read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=(0, 1), granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_cap=128, granule_cap=128, pool_fine=16, pool_rest=4,
              pool_rate=32, output_pool=128)
# Fast output synapses from the cells after the upper level.
_TAUGHT = dict(teach_after=(1,), teach_threshold=1 << 24)


def _stack(**settings):
    torch.manual_seed(0)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses.
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model.eval()


@pytest.fixture
def writer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    import generate_stream

    return generate_stream


@pytest.mark.parametrize("design", [{}, _TAUGHT])
def test_the_written_bytes_are_those_of_one_pass_over_the_whole_text(writer, design):
    model = _stack(**design)
    prompt = bytes(torch.randint(0, 256, (70,), generator=torch.Generator().manual_seed(3)).tolist())
    written = writer.continue_stream(model, prompt, 30)
    assert len(written) == 30
    text = torch.tensor([list(prompt + written)])
    with torch.no_grad():
        counts, _ = model.stream(text)
    # The count at a byte predicts the next one: every written byte is the likeliest after the text before it.
    assert counts[0, len(prompt) - 1:-1].argmax(-1).tolist() == list(written)
    assert writer.continue_stream(model, prompt, 0) == b""


def test_a_longer_reach_is_heard_while_writing(writer):
    model = _stack()
    prompt = bytes(torch.randint(0, 256, (150,), generator=torch.Generator().manual_seed(4)).tolist())

    def draw():
        return writer.continue_stream(model, prompt, 1, temperature=1.0, generator=torch.Generator().manual_seed(1))

    def counts():
        with torch.no_grad():
            return model.stream(torch.tensor([list(prompt)]))[0][0, -1]

    near = counts()
    writer.lengthen_reach(model, 140, [1])
    # The prompt is longer than the reads' window of 48 cells, so the longer reach changes what the output fires.
    assert not torch.equal(counts(), near) and len(draw()) == 1


def test_drawn_bytes_follow_the_seed(writer):
    model = _stack()

    def draw(seed, temperature):
        return writer.continue_stream(model, b"the old man", 24, temperature=temperature,
                                      generator=torch.Generator().manual_seed(seed))

    assert draw(1, 1.0) == draw(1, 1.0) != draw(2, 1.0)
