"""Echo learning: the parts of a stack learn from fixed random echoes of the error at the output, and no error is
passed back through the parts above them."""

import pytest
import torch
from torch.nn import functional as F

from snnformer.event_lm import EchoTape, EventLM, EventLMConfig, echo_feedback
from test_teaching_spikes import _STACK, _bytes


def _stack(seed=0):
    torch.manual_seed(seed)
    model = EventLM(EventLMConfig(**_STACK))
    # Reads and cell layers start silent. Give them output synapses, so that every part adds to the stream.
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for read in model.state_reads.values():
            read.project.weight.copy_(torch.randn(read.project.weight.shape, generator=generator) * 0.05)
        for layer in model.granule_layers.values():
            layer.send.copy_(torch.randn(layer.send.shape, generator=generator) * 0.05)
    return model.train()


def _loss(model, counts, tokens):
    logits = counts[:, :-1].float() * (0.6931471805599453 / (8 * model.cfg.resolution))
    return F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten())


def _reached(model):
    """The parts of the model whose parameters a backward pass reached, by the first part of their names."""
    return {name.split(".")[0] for name, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0}


def _layers(model, part):
    """The levels of ``part`` whose parameters a backward pass reached."""
    return {name.split(".")[1] for name, parameter in model.named_parameters()
            if name.startswith(part + ".") and parameter.grad is not None and parameter.grad.abs().sum() > 0}


@pytest.mark.parametrize("everywhere", [True, False])
def test_echoes_leave_what_the_stack_fires_as_it_was(everywhere):
    model, tokens = _stack(), _bytes()
    plain = model(tokens)
    tape = EchoTape(everywhere=everywhere)
    echoed = model(tokens, echo=tape)
    assert torch.equal(plain, echoed)
    names = [name for name, _ in tape.parts]
    assert names == (["input", "read0", "cells0", "read1", "cells1"] if everywhere else ["cells0", "cells1"])


def test_the_error_at_the_output_reaches_only_the_output_neurons():
    model, tokens = _stack(), _bytes()
    tape = EchoTape()
    _loss(model, model(tokens, echo=tape), tokens).backward()
    assert _reached(model) == {"output"}


def test_every_part_learns_from_its_own_echo_alone():
    model, tokens = _stack(), _bytes()
    for name, part, level in (("cells1", "granule_layers", "1"), ("read0", "state_reads", "0")):
        model.zero_grad(set_to_none=True)
        tape = EchoTape()
        model(tokens, echo=tape)
        added = dict(tape.parts)[name]
        added.backward(torch.randn_like(added))
        assert _reached(model) == {part} and _layers(model, part) == {level}
    model.zero_grad(set_to_none=True)
    tape = EchoTape()
    model(tokens, echo=tape)
    added = dict(tape.parts)["input"]
    added.backward(torch.randn_like(added))
    assert _reached(model) == {"input"}


def test_with_echoes_in_the_cells_alone_the_rest_learns_from_the_gradient():
    model, tokens = _stack(), _bytes()
    tape = EchoTape(everywhere=False)
    _loss(model, model(tokens, echo=tape), tokens).backward()
    assert {"input", "state_reads", "output"} <= _reached(model) and "granule_layers" not in _reached(model)
    # The gradient that reaches the reads and the input synapses is the one that a plain backward pass gives them
    # less what it would have taken through the cells.
    model.zero_grad(set_to_none=True)
    tape = EchoTape(everywhere=False)
    model(tokens, echo=tape)
    added = dict(tape.parts)["cells0"]
    added.backward(torch.randn_like(added))
    assert _reached(model) == {"granule_layers"} and _layers(model, "granule_layers") == {"0"}


def test_the_feedback_synapses_are_fixed_by_their_seed_and_part():
    first = echo_feedback(0, "cells1", 64)
    assert torch.equal(first, echo_feedback(0, "cells1", 64)) and first.shape == (256, 64)
    assert set(first.unique().tolist()) == {-1 / 16, 1 / 16}
    assert not torch.equal(first, echo_feedback(0, "cells0", 64))
    assert not torch.equal(first, echo_feedback(1, "cells1", 64))


def test_the_echo_update_gives_each_part_its_echo(monkeypatch):
    from test_stream_form import _trainer

    trainer, _ = _trainer(monkeypatch)
    model, tokens = _stack(), _bytes()
    # As in training: the counts at every byte of a window predict the byte after it.
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    scale = 0.6931471805599453 / (8 * model.cfg.resolution)

    def update(tape):
        logits = model(inputs, echo=tape).float() * scale
        trainer.echo_backward(F.cross_entropy(logits.flatten(0, 1), targets.flatten()), logits, targets, tape,
                              trainer.EchoFeedback(0))
        return logits

    logits = update(EchoTape())
    assert {"input", "state_reads", "granule_layers", "output"} == _reached(model)
    together = model.granule_layers["1"].send.grad.clone()
    # The echo of a part is the error of every byte, its softmax less the byte that came, through the part's
    # feedback synapses.
    model.zero_grad(set_to_none=True)
    tape = EchoTape()
    model(inputs, echo=tape)
    error = (F.softmax(logits.detach(), -1) - F.one_hot(targets, 256).float()) / targets.numel()
    added = dict(tape.parts)["cells1"]
    added.backward(error @ echo_feedback(0, "cells1", added.shape[-1]))
    assert torch.allclose(model.granule_layers["1"].send.grad, together)


@pytest.mark.parametrize("everywhere", [True, False])
def test_a_mirror_echo_tells_every_part_the_error_of_the_stream_that_the_output_hears(monkeypatch, everywhere):
    from test_stream_form import _trainer

    trainer, _ = _trainer(monkeypatch)
    model, tokens = _stack(), _bytes()
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    scale = 0.6931471805599453 / (8 * model.cfg.resolution)
    tape = EchoTape(everywhere=everywhere)
    logits = model(inputs, echo=tape).float() * scale
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    trainer.echo_backward(loss, logits, targets, tape, None, mirror=True)
    top = tape.top.grad.clone()
    assert top.shape == tape.parts[0][1].shape and top.abs().sum() > 0
    learned = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
               if name.startswith("granule_layers.1.") and parameter.grad is not None}
    assert learned
    # A part's update is its own response to that one error, as if it alone had written into the stream.
    model.zero_grad(set_to_none=True)
    tape = EchoTape(everywhere=everywhere)
    model(inputs, echo=tape)
    added = dict(tape.parts)["cells1"]
    added.backward(top)
    for name, grad in learned.items():
        assert torch.allclose(dict(model.named_parameters())[name].grad, grad, atol=1e-7)


@pytest.mark.parametrize("echo", ["all", "cells"])
def test_the_trainer_trains_a_stack_by_mirror_echoes(monkeypatch, tmp_path, echo):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / echo, "--echo", echo, "--echo-feedback", "mirror")
    assert record["args"]["echo_feedback"] == "mirror" and record["curve"][-1]["val_bpb"] > 0


def test_echoes_train_a_stack_in_windows():
    with pytest.raises(ValueError, match="echo"):
        EventLM(EventLMConfig()).train()(_bytes(), echo=EchoTape())
    model = _stack()
    with pytest.raises(ValueError, match="echo"):
        model(_bytes(), echo=EchoTape(), stream=model.stream(_bytes()[:, :4])[1])


@pytest.mark.parametrize("echo", ["all", "cells"])
def test_the_trainer_trains_a_stack_by_echoes(monkeypatch, tmp_path, echo):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / echo, "--echo", echo)
    assert record["args"]["echo"] == echo and record["args"]["echo_seed"] == 0
    saved = torch.load(tmp_path / echo / "model.pt", weights_only=True)
    rebuilt = EventLM(EventLMConfig(**saved["cfg"]))
    rebuilt.load_state_dict(saved["state_dict"])
    # The bytes of this test are random, so neither run can learn them: the echoes only take another path than the
    # gradient, and the two runs end apart.
    curve = record["curve"]
    plain = _train(trainer, monkeypatch, tmp_path / f"{echo}-plain")
    assert plain["curve"][-1]["val_bpb"] != curve[-1]["val_bpb"]


def test_the_trainer_keeps_echoes_to_stacks_in_windows(monkeypatch, tmp_path):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    with pytest.raises(SystemExit):
        _train(trainer, monkeypatch, tmp_path / "streams", "--echo", "all", "--stream", "3")


def test_a_reads_credit_paths_add_up_to_its_whole_error():
    # The error that a read passes down to the stream is the sum of what goes through its probes and what goes through
    # its tags and payloads, beside what passes it unchanged. The paths change no spike.
    model, tokens = _stack(), _bytes()
    read = model.state_reads["1"]
    stream = model.input_emit(model.input(tokens) * model.cfg.resolution).detach()
    grads, fired = {}, {}
    for credit in (None, "probe", "memory", "residual"):
        read.credit = credit
        heard = stream.clone().requires_grad_()
        out = read(heard, None, tokens)
        out.square().sum().backward()
        grads[credit], fired[credit] = heard.grad, out.detach()
    read.credit = None
    assert all(torch.equal(fired[None], value) for value in fired.values())
    assert torch.allclose(grads["probe"] + grads["memory"] - grads["residual"], grads[None], atol=1e-4)
    assert not torch.allclose(grads["probe"], grads[None]) and not torch.allclose(grads["memory"], grads[None])


def test_the_trainer_trains_with_a_reads_credit_path(monkeypatch, tmp_path):
    from test_stream_form import _train, _trainer

    trainer, _ = _trainer(monkeypatch)
    record = _train(trainer, monkeypatch, tmp_path / "probe", "--read-credit", "probe")
    assert record["args"]["read_credit"] == "probe"


def test_local_cells_learn_from_the_stream_they_write_and_pass_nothing_back():
    model, tokens = _stack(), _bytes()
    layer = model.granule_layers["1"]
    stream = model.input_emit(model.input(tokens) * model.cfg.resolution).detach()
    grads, fired, learned = {}, {}, {}
    for credit in (None, "local"):
        layer.credit = credit
        model.zero_grad(set_to_none=True)
        heard = stream.clone().requires_grad_()
        out = layer(heard, tokens)
        out.square().sum().backward()
        grads[credit], fired[credit] = heard.grad, out.detach()
        learned[credit] = layer.receive.grad.clone()
    layer.credit = None
    # The spikes and what the cells learn are as before; only the error that the cells passed back is gone, and what
    # passes the layer unchanged still carries the error below.
    assert torch.equal(fired[None], fired["local"]) and torch.allclose(learned[None], learned["local"])
    assert not torch.allclose(grads[None], grads["local"]) and grads["local"].abs().sum() > 0
