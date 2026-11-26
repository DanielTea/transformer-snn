"""The practice loop (experiments/practice_loop.py): its tasks, verifier, grader, staircase, attempts and learning."""
import random
import sys
from pathlib import Path

import pytest
import torch

from snnformer import SNNConfig
from snnformer.event_lm import EventLM, EventLMConfig
from snnformer.transformer_reference import TransformerLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import practice_loop as practice  # noqa: E402
from test_hop_tasks import _STACK  # noqa: E402


def test_a_task_is_verified_and_graded_on_its_last_byte():
    task = practice.generate(random.Random(1), 3)
    assert task.prompt.endswith(b"=") and practice.verify(task)
    digit = bytes([task.answer])
    wrong = bytes([ord("0") + (task.answer - ord("0") + 1) % 10])
    assert practice.grade(task, digit + b"\n") == 1.0
    assert practice.grade(task, b"ab=cd=" + digit + b"\n") == 1.0
    assert practice.grade(task, wrong + b"\n") == 0.0 and practice.grade(task, digit) == 0.0
    assert practice.wrote_a_name(b"ab=" + digit + b"\n") and not practice.wrote_a_name(digit + b"\n")
    broken = practice.Task(task.prompt, ord("0") + (task.answer - ord("0") + 1) % 10, task.hops)
    assert not practice.verify(broken)


def test_the_staircase_keeps_the_solve_rate_between_a_quarter_and_a_half():
    assert practice.staircase(4, 0.6) == 5 and practice.staircase(4, 0.1) == 3
    assert practice.staircase(4, 0.4) == 4 and practice.staircase(1, 0.0) == 1


def test_both_kinds_of_model_attempt_and_learn_from_disagreeing_attempts():
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    transformer = TransformerLM(SNNConfig(neuron="ann", attn="transformer", position_embeddings=True),
                                vocab=256, dim=32, depth=2, heads=4, ctx=256).eval()
    tasks = [practice.generate(random.Random(seed), 2) for seed in range(3)]
    for model in (stack, transformer):
        written = practice.attempts(model, tasks, 2, 6, 1.0, torch.Generator().manual_seed(0), "cpu")
        assert len(written) == 6 and all(0 < len(w) <= 6 for w in written)
        rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        model.train()
        loss = practice.policy_loss(model, tasks, written, rewards, 2, "cpu")
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        model.zero_grad()
        assert practice.policy_loss(model, tasks, written, torch.ones(6), 2, "cpu") is None


def test_greedy_evaluation_reports_every_hop_count():
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    found = practice.evaluate(stack, [1, 3], 4, 5, 0, "cpu")
    assert set(found) == {"1", "3"} and all(0 <= row["accuracy"] <= 1 for row in found.values())
    assert all(0 <= row["wrote_the_chain"] <= row["accuracy"] for row in found.values())


def test_an_answer_wrote_the_chain_only_if_it_writes_every_name_and_the_digit():
    task = practice.generate(random.Random(1), 3)
    steps = practice.chain_of(task.prompt.decode() + chr(task.answer) + "\n")
    right = "=".join(steps[1:]).encode() + b"\n"
    assert len(steps) == 4 and practice.wrote_the_chain(task, right) and practice.grade(task, right) == 1.0
    assert not practice.wrote_the_chain(task, bytes([task.answer]) + b"\n")
    assert not practice.wrote_the_chain(task, right[:-1]) and not practice.wrote_the_chain(task, b"zz=" + right)
    one = practice.generate(random.Random(2), 1)
    assert practice.wrote_the_chain(one, bytes([one.answer]) + b"\n")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple's GPU")
def test_both_kinds_of_model_attempt_and_learn_on_apples_gpu_which_has_no_float64():
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval().to("mps")
    transformer = TransformerLM(SNNConfig(neuron="ann", attn="transformer", position_embeddings=True),
                                vocab=256, dim=32, depth=2, heads=4, ctx=256).eval().to("mps")
    tasks = [practice.generate(random.Random(seed), 2) for seed in range(3)]
    for model in (stack, transformer):
        written = practice.attempts(model, tasks, 2, 6, 1.0, torch.Generator().manual_seed(0), "mps")
        assert len(written) == 6 and all(0 < len(w) <= 6 for w in written)
        model.train()
        loss = practice.policy_loss(model, tasks, written, torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0]), 2, "mps")
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
