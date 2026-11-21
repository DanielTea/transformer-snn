"""Four ways for a stack to hold knowledge: knowledge cells in the race of a state read, branching dendrites, apical
tufts that burst, and groups of cells whose router cells race to let them speak."""

import pytest
import torch

from snnformer.event_lm import (_BURST_LIMIT, _GRANULE_REST_LIMIT, _PACKET_UNITS, EventLM, EventLMConfig,
                                emitted_pulses)

_LEVELS = (0, 1)
# The small stack of the stream-form tests, with delayed fibres at the reads and cells as the full-size stack has them.
_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=_LEVELS, read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=_LEVELS, granule_clusters=1, granule_fan=128,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)
_KNOWING = dict(read_knowledge=8)
_BRANCHED = dict(granule_branches=4)
_APICAL = dict(granule_apical=16)
_EXPERTS = dict(granule_experts=4, granule_active=2)
_DESIGNS = [_KNOWING, _BRANCHED, _APICAL, _EXPERTS, dict(**_KNOWING, **_BRANCHED, **_APICAL, **_EXPERTS)]


def _stack(seed=0, **settings):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    # Reads and cell layers start silent. Give them output synapses, and the tufts synapses of their own.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
            if layer.apical is not None:
                layer.apical.copy_(torch.randn(layer.apical.shape, generator=generator) * 0.05)
    return model


def _bytes(seed=2, length=160, streams=2):
    return torch.randint(0, 256, (streams, length), generator=torch.Generator().manual_seed(seed))


def _pieces(model, tokens, cuts):
    counts, state, start = [], None, 0
    for length in cuts:
        fired, state = model.stream(tokens[:, start:start + length], state)
        counts.append(fired)
        start += length
    return torch.cat(counts, 1), state


def _plain_twin(model, **settings):
    """The model without the parts in ``settings``, with every synapse it shares with ``model``."""
    plain = EventLM(EventLMConfig(**dict(_STACK, **settings)))
    missing, unexpected = plain.load_state_dict(model.state_dict(), strict=False)
    assert not missing
    return plain


@pytest.mark.parametrize("design", _DESIGNS)
@pytest.mark.parametrize("cuts", [(160,), (80, 80), (47, 1, 49, 63), (1, 2, 3, 154)])
def test_a_stream_cut_anywhere_fires_the_spikes_of_one_pass(design, cuts):
    model, tokens = _stack(**design).eval(), _bytes()
    with torch.no_grad():
        whole = model(tokens)
        fired, state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole) and torch.equal(whole, whole.round()) and state.position == 160


@pytest.mark.parametrize("design", _DESIGNS)
def test_every_new_synapse_learns(design):
    model, tokens = _stack(**design).train(), _bytes()
    model(tokens).float().square().mean().backward()
    names = [name for name, parameter in model.named_parameters()
             if any(part in name for part in ("known_", "branch_rest", "apical", "router"))]
    assert names
    for name in names:
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


@pytest.mark.parametrize("design", _DESIGNS)
def test_the_sums_stay_exact(design):
    model = _stack(**design)
    model._check_accumulator()
    with torch.no_grad():
        for layer in model.granule_layers.values():
            layer.receive.mul_(400)
            if layer.apical is not None:
                layer.apical.mul_(400)
            if layer.experts > 1:
                layer.router.mul_(400)
    with pytest.raises(ValueError, match="exact FP32"):
        model._check_accumulator()
    model.synaptic_scaling()
    model._check_accumulator()


def test_knowledge_cells_join_the_race_of_every_read():
    model = _stack(**_KNOWING)
    read = model.state_reads["0"]
    width = 32 // 4
    assert read.known_tag.shape == read.known_payload.shape == (4, 8, width) and read.known_offset.shape == (4, 8)
    plain = _stack()
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * (2 * 8 * 32 + 4 * 8)
    tags, payloads, _ = read.known()
    assert torch.equal(tags, tags.round()) and torch.equal(payloads, payloads.round())
    assert tags.abs().max() <= read.fire.capped and payloads.abs().max() <= read.fire.capped
    spikes = torch.randint(-3, 4, (2, 30, 64)).float()
    with torch.no_grad():
        fired, total, payload, _ = read._address(spikes)
    # Every read reaches the 30 cells of the window and the 8 knowledge cells, and its spikes stay within the budget.
    assert fired.shape == (2, 4, 30, 38) and payload.shape == (2, 4, 38, width)
    assert fired[..., 30:].sum() > 0 and torch.all(fired[..., :30].triu(1) == 0)
    assert torch.all((fired.sum(-1) - 64).abs() <= 38)


def test_knowledge_cells_far_behind_leave_the_read_as_it_was():
    model = _stack(**_KNOWING)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.known_offset.fill_(-1e4)
    plain = _plain_twin(model)
    tokens = _bytes()
    with torch.no_grad():
        assert torch.equal(model(tokens), plain(tokens))


def test_a_branched_cell_passes_what_each_branch_holds_above_its_threshold():
    model = _stack(**_BRANCHED)
    layer = model.granule_layers["0"]
    with torch.no_grad():
        layer.branch_rest.uniform_(-2, 2)
    heard = torch.randint(-4, 5, (2, 5, 1, 128)).float()
    quanta = torch.full((2, 5, 1, 1), 4.0)
    with torch.no_grad():
        body = layer.branched(heard, quanta)
    weights = (layer.receive * 256).round().clamp(-layer.receive.new_tensor(2 ** 31), 2 ** 31)
    held = torch.einsum("btgjk,gnjk->btgnj", heard.unflatten(-1, (4, 32)), weights.unflatten(-1, (4, 32)))
    thresholds = (layer.branch_rest.clamp(-_GRANULE_REST_LIMIT, _GRANULE_REST_LIMIT) * _PACKET_UNITS).round()
    expected = (held - thresholds * 4 * layer.quarter()).clamp_min(0).sum(-1)
    assert torch.equal(body, expected) and torch.equal(body, body.round())
    plain = _stack()
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * 96 * 4


def test_a_branched_cell_tells_inputs_together_from_inputs_spread():
    model = _stack(**_BRANCHED)
    layer = model.granule_layers["0"]
    with torch.no_grad():
        # Sixteen ticks a synapse, and a branch threshold of two spikes: 512 ticks at four quanta.
        layer.receive.fill_(16 / 256)
        layer.branch_rest.fill_(2.0)
    quanta = torch.full((1, 1, 1, 1), 4.0)
    together, spread = torch.zeros(1, 1, 1, 128), torch.zeros(1, 1, 1, 128)
    together[..., :4] = 16.0
    spread[..., ::32] = 16.0
    with torch.no_grad():
        # The same charge on one branch passes its threshold; spread over four branches it passes none of them.
        assert layer.branched(together, quanta).min() > 0 and layer.branched(spread, quanta).max() == 0


def test_silent_tufts_leave_the_cells_as_they_were():
    model = _stack(**_APICAL)
    with torch.no_grad():
        for layer in model.granule_layers.values():
            layer.apical.zero_()
    plain = _plain_twin(model)
    tokens = _bytes()
    with torch.no_grad():
        assert torch.equal(model(tokens), plain(tokens))
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * 96 * 16


def test_a_tuft_multiplies_the_spikes_the_body_fires_and_fires_none_alone():
    model = _stack(**_APICAL)
    layer = model.granule_layers["0"]
    fire = layer.fire
    # A spike's threshold at four quanta is 256 ticks.
    charge = torch.tensor([[[[0.0, 256.0 * 3, 256.0 * 3, 256.0 * 200]]]])
    quanta = torch.full((1, 1, 1, 1), 4.0)
    bursts = torch.tensor([[[[3.0, 0.0, 2.0, 3.0]]]])
    counts = fire(charge, quanta, bursts=bursts)
    # Silent, three spikes, three spikes in three copies, and a full cell that cannot burst beyond its count.
    assert counts.tolist() == [[[[0.0, 3.0, 9.0, 128.0]]]]
    assert emitted_pulses(layer.burst, bursts) == 0 and emitted_pulses(fire, counts) == 140
    tuft = torch.full((1, 1, 1, 1), 256.0 * 10)
    assert layer.burst(tuft, quanta).item() == _BURST_LIMIT


def test_a_tuft_hears_its_own_level_read():
    with pytest.raises(ValueError, match="apical tuft"):
        EventLM(EventLMConfig(**dict(_STACK, read_after=(0,), **_APICAL)))
    with pytest.raises(ValueError, match="apical tuft"):
        EventLM(EventLMConfig(**dict(_STACK, granule_veto=4, **_APICAL)))


def test_router_cells_let_the_same_number_of_groups_speak_at_every_byte():
    model = _stack(**_EXPERTS).eval()
    layer = model.granule_layers["0"]
    heard = torch.randint(-4, 5, (2, 30, 1, 128)).float()
    quanta = torch.full((2, 30, 1, 1), 4.0)
    with torch.no_grad():
        opened = layer.opened(heard, quanta)
    groups = opened.unflatten(-1, (4, 24))
    assert torch.all(groups == groups[..., :1]) and torch.all(groups[..., 0].sum(-1) == 2)
    # At equal charge the earlier router cell fires first.
    with torch.no_grad():
        layer.router.zero_()
        tied = layer.opened(heard, quanta).unflatten(-1, (4, 24))[..., 0]
    assert torch.all(tied == torch.tensor([1.0, 1.0, 0.0, 0.0]))
    plain = _stack()
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in plain.parameters())
    assert extra == 2 * 4 * 128


def test_homeostasis_raises_the_threshold_of_a_router_cell_that_always_wins():
    model = _stack(**_EXPERTS).train()
    layer = model.granule_layers["0"]
    with torch.no_grad():
        layer.router.zero_()
        layer.router[:, 3] = 0.02
    heard = torch.randint(0, 5, (2, 30, 1, 128)).float()
    quanta = torch.full((2, 30, 1, 1), 4.0)
    for _ in range(3):
        layer.opened(heard, quanta)
    bias = layer.router_bias[0]
    # Group 3 wins every byte, half again its share, and its threshold climbs. The other win goes round groups 0 to
    # 2 as their thresholds sink below one another's, a third of the bytes each where their share is a half.
    assert bias[3].item() == pytest.approx(3 * 0.01 * 0.5) and bias[:3].sum().item() == pytest.approx(-3 * 0.01 * 0.5)
    assert torch.all(bias[:3] < 0.01)
    before = layer.router_bias.clone()
    with torch.no_grad():
        layer.opened(heard, quanta)
    model.eval()
    layer.opened(heard, quanta)
    assert torch.equal(layer.router_bias, before)
    assert "router_bias" in dict(model.named_buffers()).keys() or any(
        name.endswith("router_bias") for name in model.state_dict())


def test_silent_groups_send_no_pulses():
    model = _stack(**_EXPERTS).eval()
    counted = []
    for layer in model.granule_layers.values():
        layer.fire.register_forward_hook(lambda module, _inputs, counts: counted.append(counts))
    with torch.no_grad():
        model(_bytes())
    for counts in counted:
        groups = counts.unflatten(-1, (4, 24)).abs().sum(-1)
        assert torch.all((groups > 0).sum(-1) <= 2)


@pytest.mark.parametrize("settings", [dict(read_knowledge=4, read_mode="soft", read_budget=None),
                                      dict(read_knowledge=4, read_budget=None), dict(granule_branches=3),
                                      dict(granule_branches=2, granule_veto=4), dict(granule_experts=5),
                                      dict(granule_experts=4, granule_active=4), dict(granule_active=2)])
def test_settings_that_do_not_fit_are_refused(settings):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **settings)))


@pytest.mark.parametrize("growth, flags", [(dict(read_knowledge=4), ["--read-knowledge", "4"]),
                                           (dict(granule_apical=8), ["--granule-apical", "8"])])
@pytest.mark.parametrize("recipe", [[], ["--rotated-adam", "3", "--rotate-stacks"]])
def test_a_saved_run_grows_new_synapses_and_goes_on(monkeypatch, tmp_path, growth, flags, recipe):
    import sys
    from pathlib import Path

    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments" / "stack"))
    import fork_state

    assert _train(trainer, monkeypatch, tmp_path / "parent", *recipe, "--stop-at", "6") is None
    parent = torch.load(tmp_path / "parent" / "model.pt.resume", weights_only=True)
    forked = fork_state.fork(parent, "grown", growth)
    # All that the run learned is kept: its synapses, their running averages and the optimizer's moments.
    assert all(torch.equal(forked["state_dict"][key], value) for key, value in parent["state_dict"].items())
    new = sorted(set(forked["state_dict"]) - set(parent["state_dict"]))
    assert new and all(("known_" in key) or ("apical" in key) for key in new)
    assert len(forked["average"]) > len(parent["average"])
    old_model, _, old_order = fork_state._models(parent["args"])
    model, _, order = fork_state._models(forked["args"])
    for index, name in enumerate(order):
        if name in old_order and old_order.index(name) in parent["optimizer"]["state"]:
            kept = forked["optimizer"]["state"][index]
            saved = parent["optimizer"]["state"][old_order.index(name)]
            assert all(torch.equal(kept[key], saved[key]) for key in saved if torch.is_tensor(saved[key]))
        elif name not in old_order:
            assert index not in forked["optimizer"]["state"]
    # The grown stack starts as the parent's: silent tufts change nothing, knowledge cells far behind next to nothing.
    old_model.load_state_dict(parent["state_dict"])
    model.load_state_dict(forked["state_dict"])
    tokens = torch.randint(0, 256, (2, 48), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        before, after = old_model.eval()(tokens), model.eval()(tokens)
    changed = (before != after).any(-1).float().mean().item()
    assert changed == 0 if "granule_apical" in growth else changed < 0.1
    (tmp_path / "child").mkdir()
    torch.save(forked, tmp_path / "child" / "model.pt.resume")
    record = _train(trainer, monkeypatch, tmp_path / "child", *recipe, *flags, "--tag", "grown")
    assert record["resumed_at_steps"] == [6]
    final = torch.load(tmp_path / "child" / "model.pt", weights_only=True)["state_dict"]
    assert any(not torch.equal(final[key], forked["state_dict"][key]) for key in new if final[key].is_floating_point())
    assert sys.modules["fork_state"] is fork_state


_LIBRARY = dict(read_knowledge=8, read_knowledge_shared=True)


def test_a_library_of_knowledge_cells_serves_every_level():
    model = _stack(**_LIBRARY)
    reads = list(model.state_reads.values())
    assert reads[0].known_tag is reads[1].known_tag and reads[0].known_payload is reads[1].known_payload
    assert reads[0].known_offset is not reads[1].known_offset
    plain = _stack()
    extra = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in plain.parameters())
    # One library of tags and payloads, and offsets for every level.
    assert extra == 2 * 8 * 32 + 2 * 4 * 8
    # Both levels' errors reach the shared synapses.
    tokens = _bytes()
    model.train()(tokens).float().square().mean().backward()
    shared = reads[0].known_tag.grad.clone()
    model.zero_grad()
    with torch.no_grad():
        model.state_reads["1"].project.weight.zero_()
    model(tokens).float().square().mean().backward()
    assert reads[0].known_tag.grad is not None and not torch.equal(reads[0].known_tag.grad, shared)


@pytest.mark.parametrize("cuts", [(160,), (47, 1, 49, 63)])
def test_a_library_streams_as_one_pass(cuts):
    model, tokens = _stack(**_LIBRARY).eval(), _bytes()
    with torch.no_grad():
        whole = model(tokens)
        fired, _ = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole)


def test_a_library_needs_knowledge_cells():
    with pytest.raises(ValueError, match="library"):
        EventLM(EventLMConfig(**dict(_STACK, read_knowledge_shared=True)))


@pytest.mark.parametrize("cuts", [(160,), (47, 1, 49, 63), (1, 2, 3, 154)])
def test_fast_synapses_keyed_by_knowledge_learn_the_same_however_a_stream_is_cut(cuts):
    model, tokens = _stack(**_KNOWING).eval(), _bytes()
    model.teach((0, 1), 1 << 22, knowledge=True)
    with torch.no_grad():
        fired, state = _pieces(model, tokens, (160,))
        cut, cut_state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, cut) and torch.equal(state.taught.fast, cut_state.taught.fast)
    # Every taught level adds its cells and the knowledge cells of its read's heads to what the fast synapses hear.
    assert state.taught.fast.shape[-1] == 2 * (96 + 4 * 8)


def test_knowledge_fibres_change_what_the_fast_synapses_learn():
    model, tokens = _stack(**_KNOWING).eval(), _bytes()
    with torch.no_grad():
        model.teach((0, 1), 1 << 22)
        plain, plain_state = _pieces(model, tokens, (160,))
        model.teach((0, 1), 1 << 22, knowledge=True)
        keyed, keyed_state = _pieces(model, tokens, (160,))
        model.teach(None)
        untaught, _ = _pieces(model, tokens, (160,))
    assert plain_state.taught.fast.shape[-1] < keyed_state.taught.fast.shape[-1]
    assert not torch.equal(plain, keyed) and not torch.equal(keyed, untaught)
    assert torch.equal(keyed, keyed.round())


@pytest.mark.parametrize("settings", [dict(teach_knowledge=True), dict(teach_after=(0,), teach_threshold=1 << 22,
                                                                         teach_knowledge=True)])
def test_knowledge_fibres_need_taught_levels_with_knowledge_cells(settings):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **settings)))


_KEEPING = dict(read_kept=6)


@pytest.mark.parametrize("design", [_KEEPING, dict(**_KEEPING, **_KNOWING)])
@pytest.mark.parametrize("cuts", [(200,), (47, 1, 49, 103), (64, 64, 72), (1, 2, 3, 194), (130, 70)])
def test_kept_cells_fire_the_spikes_of_one_pass_however_a_stream_is_cut(design, cuts):
    model, tokens = _stack(**design).eval(), _bytes(length=200)
    with torch.no_grad():
        whole, state = _pieces(model, tokens, (200,))
        fired, cut_state = _pieces(model, tokens, cuts)
    assert torch.equal(fired, whole) and torch.equal(whole, whole.round())
    for key, tape in state.reads.items():
        other = cut_state.reads[key]
        assert torch.equal(tape.kept_score, other.kept_score) and torch.equal(tape.kept_tag, other.kept_tag)


def test_a_stream_byte_at_a_time_keeps_the_same_cells():
    model, tokens = _stack(**_KEEPING).eval(), _bytes(length=140)
    with torch.no_grad():
        whole, state = _pieces(model, tokens, (140,))
        single, single_state = _pieces(model, tokens, (1,) * 140)
    assert torch.equal(whole, single)
    assert all(torch.equal(tape.kept_score, single_state.reads[key].kept_score) for key, tape in state.reads.items())


def test_cells_that_left_the_reach_fill_the_slots_with_whole_pulses():
    model, tokens = _stack(**_KEEPING).eval(), _bytes(length=200)
    with torch.no_grad():
        _, state = _pieces(model, tokens, (200,))
    for tape in state.reads.values():
        # The reach is 48: by byte 192 three changes have let the cells that left it compete for 6 slots a head.
        assert tape.kept_tag.shape == (2, 4, 6, 8) and (tape.kept_score >= 0).sum() > 0
        assert torch.equal(tape.kept_tag, tape.kept_tag.round()) and torch.equal(tape.kept_payload, tape.kept_payload.round())
        assert torch.equal(tape.kept_score, tape.kept_score.round())
        # The tape holds the cells within reach and those that left it since the last change, 200 - 192 + 47 of them.
        assert tape.payload.shape[2] == 47 + 200 - 192


def test_silent_kept_cells_leave_the_read_as_it_was():
    model = _stack(**_KEEPING)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.kept_offset.fill_(-1e4)
    plain = _plain_twin(model)
    tokens = _bytes(length=200)
    with torch.no_grad():
        kept, _ = _pieces(model.eval(), tokens, (73, 127))
        reference, _ = _pieces(plain.eval(), tokens, (73, 127))
    assert torch.equal(kept, reference)


def test_a_state_with_kept_cells_serves_any_number_of_continuations():
    model, tokens = _stack(**_KEEPING).eval(), _bytes(length=200)
    with torch.no_grad():
        _, state = model.stream(tokens[:, :130])
        saved = {key: (tape.kept_score.clone(), tape.importance.clone()) for key, tape in state.reads.items()}
        first, _ = model.stream(tokens[:, 130:], state)
        second, _ = model.stream(tokens[:, 130:], state)
    assert torch.equal(first, second)
    for key, tape in state.reads.items():
        assert torch.equal(tape.kept_score, saved[key][0]) and torch.equal(tape.importance, saved[key][1])


def test_the_most_important_cells_take_the_slots():
    read = _stack(**dict(_KEEPING, read_kept=2)).state_reads["0"]
    kept_score = torch.tensor([[[5.0, -1.0]]])
    importance = torch.tensor([[[3.0, 0.0, 7.0]]])
    tags = torch.arange(3.0).view(1, 1, 3, 1).expand(1, 1, 3, 8)
    kept_tags = torch.full((1, 1, 2, 8), 9.0)
    new_tag, _, new_score = read._keep(kept_tags, kept_tags, kept_score, tags, tags, importance)
    # The newcomer with 7 spikes and the kept cell with 5 (faded by a 128th, rounded down) take the two slots.
    assert new_score.tolist() == [[[7.0, 5.0]]] and new_tag[0, 0, :, 0].tolist() == [2.0, 9.0]
    # At equal importance a kept cell stays ahead of a newcomer.
    _, _, tied = read._keep(kept_tags, kept_tags, torch.tensor([[[3.0, -1.0]]]), tags, tags,
                            torch.tensor([[[3.0, 0.0, 0.0]]]))
    assert tied.tolist() == [[[3.0, 3.0]]]


def test_kept_cells_learn_when_trained_on_streams():
    model, tokens = _stack(**_KEEPING).train(), _bytes(length=200)
    _, state = model.stream(tokens[:, :160])
    model.zero_grad()
    fired, _ = model.stream(tokens[:, 160:], state)
    fired.float().square().mean().backward()
    grads = [read.kept_offset.grad for read in model.state_reads.values()]
    assert all(grad is not None and grad.abs().sum() > 0 for grad in grads)


@pytest.mark.parametrize("settings", [dict(read_kept=-1), dict(read_kept=4, read_budget=None),
                                      dict(read_kept=4, read_mode="soft", read_budget=None)])
def test_kept_cells_need_a_race_with_a_budget(settings):
    with pytest.raises(ValueError):
        EventLM(EventLMConfig(**dict(_STACK, **settings)))
