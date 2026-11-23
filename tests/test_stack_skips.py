"""Skip fibres: a level's read and cells also hear the stream as it stood at the input and after every earlier
level, each through a gain of its own for every channel."""

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from snnformer.event_lm import _FP32_EXACT, _SKIP_LIMIT, _SKIP_UNITS, EventLM, EventLMConfig, _mixed_charge, _skip_gains
from test_teaching_spikes import _STACK, _bytes, _pieces

_DEEP = dict(_STACK, depth=3, read_after=(0, 1, 2), granule_after=(0, 1, 2))
# The released stack's options: cells and tags that hear every channel again one to three bytes later, and reads
# that learn a forgetting slope and reach further in a stream than their window.
_DELAYED = dict(granule_fan=128, granule_delays=4, read_delays=4, read_forget=True, read_reach=96)


def _stack(seed=0, dtype=torch.float32, gains=0.0, **settings):
    """A stack whose reads and cells write into the stream, with skip gains of about ``gains`` in size."""
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_DEEP, **settings)))
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in dict.fromkeys(model.state_reads.values()):
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in dict.fromkeys(model.granule_layers.values()):
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
        if gains:
            for gain in model.skips.values():
                gain.copy_(torch.randn(gain.shape, generator=generator) * gains)
    return model.to(dtype).eval()


def test_silent_fibres_leave_the_stack_as_it_is():
    plain, tokens = _stack(), _bytes()
    for mode, mixes in (("level", 1), ("parts", 3)):
        model = _stack(stack_skips=mode)
        # The first read's stream is the input, so it has no fibres. A later read hears the input and the stream
        # after every level but the last before it; the cells hear that one too.
        assert {name: tuple(gain.shape) for name, gain in model.skips.items()} == {
            "cells0": (64,), "read1": (mixes * 64,), "cells1": (2 * 64,), "read2": (mixes * 2 * 64,),
            "cells2": (3 * 64,)}
        count = lambda stack: sum(parameter.numel() for parameter in stack.parameters())  # noqa: E731
        assert count(model) == count(plain) + (6 + 3 * mixes) * 64
        with torch.no_grad():
            assert torch.equal(model(tokens), plain(tokens))
        assert torch.equal(_pieces(model, tokens, (7,) * 22 + (6,))[0], _pieces(plain, tokens, (160,))[0])


def test_the_mixed_charge_is_the_exact_sum_to_the_nearest_tick():
    generator = torch.Generator().manual_seed(0)
    ticks = torch.randint(-127, 128, (40, 64), generator=generator).float().requires_grad_()
    # Counts of up to 1,500 spikes in sixteenths: a stream's whole spikes and what skip fibres add.
    heard = (torch.randint(-24000, 24001, (3, 50, 64), generator=generator).float() / _SKIP_UNITS).requires_grad_()
    charge = _mixed_charge(lambda counts: F.linear(counts, ticks), heard)
    exact = F.linear(heard.detach().double(), ticks.detach().double())
    # To the nearest tick, a half tick upward: the same however the counts are split into whole spikes and a rest.
    assert not torch.equal(exact, exact.round()) and ((exact - exact.floor()) == 0.5).any()
    assert torch.equal(charge.detach().double(), (exact + 0.5).floor())
    other = heard.detach().floor()
    rest = F.linear((heard.detach() - other) * _SKIP_UNITS, ticks) / _SKIP_UNITS
    split = F.linear(other, ticks) + (rest + 0.5).floor()
    assert torch.equal(split, charge.detach())
    # Whole spikes charge as a stream's do, with nothing left for the branch.
    whole = heard.detach().round()
    assert torch.equal(_mixed_charge(lambda counts: F.linear(counts, ticks), whole), F.linear(whole, ticks))
    # The charge learns as the unrounded sum does.
    weights = torch.randn(charge.shape, generator=generator)
    (charge * weights).sum().backward()
    assert torch.allclose(heard.grad, F.linear(weights, ticks.detach().T))
    assert torch.allclose(ticks.grad, torch.einsum("bto,bti->oi", weights, heard.detach()), rtol=1e-4, atol=1e-2)


def test_gains_are_whole_sixteenths_that_share_a_largest_size():
    stored = torch.tensor([[[0.03, 0.26, -1.0], [0.04, -0.2, -3.0]]], requires_grad=True)
    steps = _skip_gains(stored)
    # A gain rounds to its nearest sixteenth. The third channel's fibres hold four copies and shrink to two.
    assert torch.equal(steps.detach(), torch.tensor([[[0.0, 4.0, -8.0], [1.0, -3.0, -24.0]]]))
    assert steps.detach().abs().sum(1).max() <= _SKIP_LIMIT
    steps.sum().backward()
    assert torch.allclose(stored.grad[..., :2], torch.full((1, 2, 2), float(_SKIP_UNITS)))


@pytest.mark.parametrize("mode", ["level", "parts"])
def test_fibres_change_what_the_levels_above_the_first_read_fire(mode):
    plain, model, tokens = _stack(), _stack(stack_skips=mode, gains=0.3), _bytes()
    with torch.no_grad():
        assert not torch.equal(model(tokens), plain(tokens))
        # With the gains of one part alone the stack differs too: each part hears its fibres.
        for name in model.skips:
            alone = _stack(stack_skips=mode)
            alone.skips[name].copy_(model.skips[name])
            assert not torch.equal(alone(tokens), plain(tokens)), name


def test_the_parts_of_a_read_hear_their_own_fibres():
    model, tokens = _stack(stack_skips="parts", **_DELAYED), _bytes()
    with torch.no_grad():
        plain = model(tokens)
        for part in range(3):
            # One gain on the input's fibres to the probes, to the tags or to the payloads of the second read.
            model.skips["read1"].zero_()
            model.skips["read1"].view(3, 1, -1)[part] = 0.5
            assert not torch.equal(model(tokens), plain), part


@pytest.mark.parametrize("settings", [
    dict(stack_skips="level"),
    dict(stack_skips="parts"),
    dict(stack_skips="level", **_DELAYED),
    dict(stack_skips="parts", read_delay_probes=True, **_DELAYED),
    dict(stack_skips="parts", read_kept=6, **_DELAYED),
    dict(stack_skips="parts", depth=2, read_after=(0, 1), granule_after=(0, 1), stack_loop=(0, 1, 2), **_DELAYED),
])
def test_a_stream_with_fibres_fires_what_its_window_fires_however_it_is_cut(settings):
    model, tokens = _stack(dtype=torch.float64, gains=0.3, **settings), _bytes(length=200)
    whole, state = _pieces(model, tokens, (200,))
    if not settings.get("read_kept"):
        # A read with kept cells fires in a stream what it cannot in a window, where it keeps none.
        with torch.no_grad():
            assert torch.equal(model(tokens), whole)
    for cuts in ((1,) * 200, (7,) * 28 + (4,), (128, 72)):
        assert torch.equal(_pieces(model, tokens, cuts)[0], whole)
    if settings.get("read_delay_probes"):
        # The probes' late fibres bring what the probes heard, which is not what the tags heard.
        tape = state.reads["1"]
        assert tape.late_probe.shape == tape.late.shape and not torch.equal(tape.late_probe, tape.late)
    # The delayed fibres bring sixteenths of spikes, not the stream's whole ones.
    if settings.get("granule_delays"):
        late = state.cells["1"].late
        assert torch.equal(late * _SKIP_UNITS, (late * _SKIP_UNITS).round()) and not torch.equal(late, late.round())


def test_every_pass_of_a_loop_has_its_own_fibres():
    settings = dict(depth=2, read_after=(0, 1), granule_after=(0, 1), stack_loop=(0, 1, 3))
    plain, model = _stack(**settings), _stack(stack_skips="level", **settings)
    assert sorted(model.skips) == sorted(["cells0"] + [f"{part}{level}" for level in range(1, 6)
                                                       for part in ("read", "cells")])
    assert model.state_reads["4"] is model.state_reads["0"]
    count = lambda stack: sum(parameter.numel() for parameter in stack.parameters())  # noqa: E731
    # The read of pass p hears p streams and its cells one more.
    assert count(model) == count(plain) + sum(level + level + 1 for level in range(6)) * 64
    tokens = _bytes()
    with torch.no_grad():
        silent = model(tokens)
        assert torch.equal(silent, plain(tokens))
        # A gain of the last pass alone: the passes fire with the same synapses and hear different fibres.
        model.skips["read4"].view(1, 4, -1)[0, 0] = 0.5
        assert not torch.equal(model(tokens), silent)


def test_the_gains_and_the_levels_below_learn_through_the_fibres():
    model, tokens = _stack(stack_skips="parts", gains=0.2).train(), _bytes()
    model(tokens).square().mean().backward()
    for name, gain in model.skips.items():
        assert gain.grad is not None and gain.grad.abs().sum() > 0, name
    # Silent fibres learn as well: a gain of zero has a slope.
    silent = _stack(stack_skips="level").train()
    silent(tokens).square().mean().backward()
    assert all(gain.grad.abs().sum() > 0 for gain in silent.skips.values())


def test_scaling_keeps_every_sum_exact_where_fibres_bring_channels_again():
    model = _stack(stack_skips="parts", gains=0.6, **_DELAYED)
    limit = _FP32_EXACT // model.input_emit.cap
    read, layer = model.state_reads["2"], model.granule_layers["2"]
    with torch.no_grad():
        # Synapses that fill the range of a stack without fibres.
        for synapse in (read.probe, read.tag, read.payload):
            synapse.weight.mul_(0.9 * limit / (synapse.weight.abs().sum(-1, keepdim=True) * 256))
        layer.receive.mul_(0.5 * limit / (layer.receive.abs().sum(-1, keepdim=True) * 256))
    silent = _stack(stack_skips="parts", **_DELAYED)
    silent.load_state_dict({name: value for name, value in model.state_dict().items() if not name.startswith("skips.")},
                           strict=False)
    silent._check_accumulator()
    with pytest.raises(ValueError, match="exact"):
        model._check_accumulator()
    reaches = model._skip_reaches()
    # A fibre's reach is the stream's own count and the sizes of the gains on its channel.
    assert reaches[read].shape == (3, 64) and reaches[layer].shape == (64,)
    assert reaches[read].min() >= 1 and 1.5 < reaches[read].max() <= 1 + (_SKIP_LIMIT + 1) / _SKIP_UNITS
    assert model.synaptic_scaling() > 0
    model._check_accumulator()
    for synapse, reach in zip((read.probe, read.tag, read.payload), read._reaches(reaches[read])):
        assert (synapse.weight_ticks().abs() * reach).sum(-1).max() <= limit
    # The scaled stack runs, and a second pass of the rule leaves the synapses where the first put them.
    with torch.no_grad():
        model(_bytes())
    placed = read.tag.weight.clone()
    model.synaptic_scaling()
    assert torch.allclose(read.tag.weight, placed, rtol=1e-5, atol=0)
    # The same synapses with silent fibres keep their sizes.
    kept = silent.state_reads["2"].probe.weight.clone(), silent.granule_layers["2"].receive.clone()
    assert silent.synaptic_scaling() == 0
    assert torch.equal(silent.state_reads["2"].probe.weight, kept[0])
    assert torch.equal(silent.granule_layers["2"].receive, kept[1])


def test_a_saved_stack_with_fibres_is_rebuilt_from_its_configuration():
    model, tokens = _stack(stack_skips="parts", gains=0.3), _bytes()
    assert model.cfg.stack_skips == "parts"
    rebuilt = EventLM(EventLMConfig(**asdict(model.cfg)))
    rebuilt.load_state_dict(model.state_dict())
    with torch.no_grad():
        assert torch.equal(rebuilt.eval()(tokens), model(tokens))


@pytest.mark.parametrize("settings", [
    dict(stack_skips="all"), dict(stack_skips="level", granule_gate=4), dict(stack_skips="level", granule_traces=1),
    dict(stack_skips="level", granule_branches=2), dict(stack_skips="parts", read_mode="soft"),
    dict(stack_skips="level", granule_apical=16),
])
def test_fibres_reach_race_reads_and_plain_cells(settings):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_DEEP, **settings)))


def test_fibres_need_a_stack():
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(stack_skips="level"))


def test_the_trainer_trains_a_stack_with_fibres(monkeypatch, tmp_path):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / "skips", "--stack-skips", "parts", "--stream", "3")
    assert record["args"]["stack_skips"] == "parts" and record["cfg"]["stack_skips"] == "parts"
    saved = torch.load(tmp_path / "skips" / "model.pt", weights_only=True)
    rebuilt = EventLM(EventLMConfig(**saved["cfg"]))
    rebuilt.load_state_dict(saved["state_dict"])
    assert rebuilt.skips is not None and all(gain.ndim == 1 for gain in rebuilt.skips.values())
    # The gains moved from zero, and a gain steps without decay: it is one of a row, not a synapse of a matrix.
    assert any(gain.abs().sum() > 0 for gain in rebuilt.skips.values())
    monkeypatch.syspath_prepend(str(trainer.Path(trainer.__file__).parent))
    from common import make_optimizer

    optimizer = make_optimizer(rebuilt, 1e-3, 0.05)
    gains = {id(gain) for gain in rebuilt.skips.values()}
    for group in optimizer.param_groups:
        if any(id(parameter) in gains for parameter in group["params"]):
            assert group["weight_decay"] == 0
