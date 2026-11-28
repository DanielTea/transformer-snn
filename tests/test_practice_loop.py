"""The practice loop (experiments/practice_loop.py): its tasks, verifier, grader, staircase, attempts and learning."""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

from snnformer import SNNConfig
from snnformer.event_lm import EventLM, EventLMConfig
from snnformer.transformer_reference import TransformerLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import hop_tasks  # noqa: E402
import practice_loop as practice  # noqa: E402
import task_families  # noqa: E402
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


def test_the_staircase_stops_at_its_longest_chain():
    assert practice.staircase(8, 0.9, longest=8) == 8 and practice.staircase(7, 0.9, longest=8) == 8
    assert practice.staircase(8, 0.1, longest=8) == 7 and practice.staircase(24, 0.9) == 24


def test_an_evaluation_along_the_way_is_logged_and_its_model_kept(tmp_path):
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).train()
    before = {name: value.clone() for name, value in stack.state_dict().items()}
    rows = [practice.keep_evaluation(stack, {"format_version": 1}, tmp_path, round_, 4, [1, 3], 4, 5, 7, "cpu")
            for round_ in (0, 50)]
    assert stack.training and all(torch.equal(before[name], value) for name, value in stack.state_dict().items())
    logged = [json.loads(line) for line in (tmp_path / "evaluations.jsonl").read_text().splitlines()]
    assert logged == rows and [row["round"] for row in logged] == [0, 50] and set(rows[0]["by_hops"]) == {"1", "3"}
    assert rows[0]["by_hops"] == practice.evaluate(stack.eval(), [1, 3], 4, 5, 7, "cpu")
    kept = torch.load(tmp_path / "model-r00050.pt", map_location="cpu", weights_only=False)
    assert kept["practice_rounds"] == 50 and kept["practice_hops"] == 4 and kept["format_version"] == 1
    assert all(torch.equal(before[name], value) for name, value in kept["state_dict"].items())


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


def test_without_dips_only_the_rewarded_attempts_teach():
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    tasks = [practice.generate(random.Random(seed), 2) for seed in range(2)]
    written = practice.attempts(stack, tasks, 4, 6, 1.0, torch.Generator().manual_seed(0), "cpu")
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    stack.train()

    def gradient(rewards_, dips):
        stack.zero_grad()
        loss = practice.policy_loss(stack, tasks, written, rewards_, 4, "cpu", dips)
        loss.backward()
        return float(loss.detach()), torch.cat([p.grad.flatten() for p in stack.parameters() if p.grad is not None])

    grpo, grpo_gradient = gradient(rewards, 1.0)
    assert grpo == gradient(rewards, 1.0)[0]
    assert grpo == float(practice.policy_loss(stack, tasks, written, rewards, 4, "cpu").detach())
    bursts, bursts_gradient = gradient(rewards, 0.0)
    halves, halves_gradient = gradient(rewards, 0.5)
    # the loss is linear in the dips' weight: GRPO is the bursts plus the dips
    assert halves == pytest.approx((grpo + bursts) / 2, abs=1e-5)
    assert torch.allclose(halves_gradient, (grpo_gradient + bursts_gradient) / 2, atol=1e-5)
    assert not torch.allclose(bursts_gradient, grpo_gradient, atol=1e-5)
    # without dips the loss is the rewarded attempts' likelihood alone: one lone reward among four attempts
    # (an advantage of sqrt(3)) and three among four (1 / sqrt(3) each), over all eight rows
    with torch.no_grad():
        texts = [tasks[i // 4].prompt + written[i] for i in range(8)]
        width = max(map(len, texts))
        tokens = torch.tensor([[practice.NEWLINE] * (width - len(t)) + list(t) for t in texts])
        logits = practice.scores(stack, stack(tokens[:, :-1])).float()
        logp = torch.log_softmax(logits, -1).gather(2, tokens[:, 1:, None])[..., 0]
        mean = torch.stack([logp[i, width - 1 - len(written[i]):].mean() for i in range(8)])
    weight = torch.tensor([3 ** 0.5, 0, 0, 0, 0, 3 ** -0.5, 3 ** -0.5, 3 ** -0.5])
    assert bursts == pytest.approx(float(-(weight * mean).mean()), abs=1e-4)


def test_rehearsal_asks_for_the_chain_as_the_corpus_writes_it():
    import hop_tasks

    rng = random.Random(4)
    lines = [hop_tasks.example(rng, hops) for hops in (1, 2, 4)]
    tasks = [practice.Task(line[:line.rindex("=") + 1].encode(), ord(line[line.rindex("=") + 1]), hops)
             for line, hops in zip(lines, (1, 2, 4))]
    for line, task in zip(lines, tasks):
        # the corpus's line with its chain written is the task's prompt followed by what rehearsal asks for
        assert hop_tasks.write_chain(line).encode() == task.prompt + practice.chain_written(task)
        assert practice.wrote_the_chain(task, practice.chain_written(task))
    assert practice.chain_written(tasks[0]) == bytes([tasks[0].answer]) + b"\n"
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).train()
    loss = practice.rehearsal_loss(stack, tasks, "cpu")
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stack.parameters())
    # the loss is the mean over the tasks of the mean cost a byte of the chain, each task read as one window
    with torch.no_grad():
        texts = [task.prompt + practice.chain_written(task) for task in tasks]
        width = max(map(len, texts))
        tokens = torch.tensor([[practice.NEWLINE] * (width - len(t)) + list(t) for t in texts])
        logits = practice.scores(stack, stack(tokens[:, :-1])).float()
        logp = torch.log_softmax(logits, -1).gather(2, tokens[:, 1:, None])[..., 0]
        costs = [-logp[row, width - 1 - len(practice.chain_written(task)):].mean() for row, task in enumerate(tasks)]
    assert float(loss.detach()) == pytest.approx(float(torch.stack(costs).mean()), abs=1e-4)
    # a model that already writes the chain has nothing to rehearse: the written bytes are all that is asked
    assert practice.written_likelihood(stack, [tasks[0].prompt], [b""], "cpu").item() == 0.0


def test_greedy_evaluation_reports_every_hop_count():
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    found = practice.evaluate(stack, [1, 3], 4, 5, 0, "cpu")
    assert set(found) == {"1", "3"} and all(0 <= row["accuracy"] <= 1 for row in found.values())
    assert all(0 <= row["wrote_the_chain"] <= row["accuracy"] for row in found.values())


def test_an_answer_wrote_the_chain_only_if_it_writes_every_name_and_the_digit():
    task = practice.generate(random.Random(1), 3)
    steps = hop_tasks.chain_of(task.prompt.decode() + chr(task.answer) + "\n")
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


def test_the_speed_options_are_refused_without_a_cuda_device(tmp_path, monkeypatch, capsys):
    for option in ("--exact-products", "--fused-reads"):
        monkeypatch.setattr(sys, "argv", ["practice_loop.py", "--checkpoint", str(tmp_path / "none.pt"), "--out",
                                          str(tmp_path / "out"), "--device", "cpu", option])
        with pytest.raises(SystemExit):
            practice.main()
        assert "take a CUDA device" in capsys.readouterr().err


@pytest.mark.parametrize("family", ["hops", "offs", "sums", "tots", "ticks"])
def test_every_family_gives_tasks_the_loop_can_verify_grade_and_rehearse(family):
    task = practice.generate(random.Random(3), 3, family)
    assert task.family == family and task.hops == 3 and task.prompt.endswith(b"=") and practice.verify(task)
    right = practice.chain_written(task)
    line = task.prompt.decode() + chr(task.answer) + "\n"
    assert right == "=".join(task_families.FAMILIES[family].chain(line)).encode() + b"\n"
    assert right[-2] == task.answer and right.count(b"=") == 2 + (family == "ticks")  # ticks end with a step more
    assert practice.wrote_the_chain(task, right) and practice.grade(task, right) == 1.0
    assert not practice.wrote_the_chain(task, bytes([task.answer]) + b"\n")
    wrong = practice.Task(task.prompt, ord("0") + (task.answer - ord("0") + 1) % 10, task.hops, family)
    assert not practice.verify(wrong) and not practice.verify(practice.Task(task.prompt, task.answer, 2, family))


def test_the_default_family_draws_the_hop_tasks_it_always_drew():
    ours, theirs = random.Random(11), random.Random(11)
    for hops in (1, 4, 9):
        task, line = practice.generate(ours, hops), hop_tasks.example(theirs, hops)
        assert task.prompt.decode() + chr(task.answer) + "\n" == line and task.family == "hops"
        assert practice.chain_written(task) == hop_tasks.write_chain(line)[len(task.prompt):].encode()


def test_an_evaluation_of_several_families_holds_each(tmp_path):
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    row = practice.keep_evaluation(stack, {}, tmp_path, 3, {"hops": 4, "sums": 5}, [1, 2], 3, 5, 0, "cpu",
                                   ["hops", "sums"])
    assert row["hops"] == {"hops": 4, "sums": 5} and set(row["by_family"]) == {"hops", "sums"}
    assert all(set(found) == {"1", "2"} for found in row["by_family"].values())
    assert json.loads((tmp_path / "evaluations.jsonl").read_text()) == row
    assert practice.evaluate(stack, [2], 3, 5, 0, "cpu", "sums") == {"2": row["by_family"]["sums"]["2"]}


def test_a_family_s_own_cap_must_name_a_practised_family(tmp_path, monkeypatch, capsys):
    for pair in ("sums=12", "hops=0", "hops=x", "hops"):
        monkeypatch.setattr(sys, "argv", ["practice_loop.py", "--checkpoint", str(tmp_path / "none.pt"), "--out",
                                          str(tmp_path / "out"), "--families", "hops", "sums", "--practise", "hops",
                                          "--longest", pair])
        with pytest.raises(SystemExit):
            practice.main()
        assert "--longest" in capsys.readouterr().err


def test_replay_keeps_only_paid_whole_chains_past_the_rehearsed_lengths():
    rng = random.Random(5)
    short, long_ = practice.generate(rng, 3), practice.generate(rng, 6)
    replay = practice.Replay(keep=3, rehearsed=[1, 2, 3, 4], seed=1)
    right, wrong = practice.chain_written(long_), b"zz=1\n"
    # two attempts a task: the short task's whole chain is within the rehearsed lengths, so it is not kept
    assert replay.add([short, long_], [practice.chain_written(short), wrong, right, right], 2) == 2
    assert replay.kept == [long_, long_]
    for _ in range(3):
        replay.add([long_], [right, wrong], 2)
    assert len(replay.kept) == 3  # the newest three
    drawn = replay.draw(2)
    assert len(drawn) == 2 and all(task is long_ for task in drawn)
    assert practice.Replay(keep=3, rehearsed=[1, 2], seed=1).draw(4) == []
    again = practice.Replay(keep=3, rehearsed=[1, 2, 3, 4], seed=1)
    again.kept = list(replay.kept)
    assert [id(t) for t in again.draw(2)] == [id(t) for t in practice.Replay(3, [4], 1).rng.sample(replay.kept, 2)]


def _tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK))
    path = tmp_path / "base.pt"
    torch.save({"task": "event_lm", "model_type": "EventLM", "cfg": dict(_STACK), "state_dict": stack.state_dict()},
               path)
    return path


def _run_loop(tmp_path, monkeypatch, out, *flags):
    monkeypatch.setattr(sys, "argv", ["practice_loop.py", "--checkpoint", str(_tiny_checkpoint(tmp_path)),
                                      "--out", str(tmp_path / out), "--rounds", "2", "--tasks", "2",
                                      "--attempts", "2", "--max-bytes", "6", "--start-hops", "6",
                                      "--rehearse", "2", *flags])
    practice.main()
    return [json.loads(line) for line in (tmp_path / out / "rounds.jsonl").read_text().splitlines()]


def test_replay_in_the_loop_rehearses_kept_chains_and_changes_nothing_when_off(tmp_path, monkeypatch):
    plain = _run_loop(tmp_path, monkeypatch, "plain")
    assert not any(key.startswith("replay") for row in plain for key in row)
    # a model this small never writes a whole chain, so the test pays every attempt that reaches the buffer
    monkeypatch.setattr(practice.Replay, "add", lambda self, tasks, written, count: (
        self.kept.extend(tasks), len(tasks))[1])
    rows = _run_loop(tmp_path, monkeypatch, "replay", "--replay", "2")
    assert rows[0]["replay_added"] == rows[0]["tasks"] and rows[0]["replay_kept"] == rows[0]["tasks"]
    assert rows[0]["replayed"] == 2 and rows[0]["replay_loss"] > 0
    assert rows[1]["replay_kept"] == rows[0]["tasks"] + rows[1]["tasks"]
    # the practice's own tasks are drawn as without replay: the first round's example is the same
    assert rows[0]["example"][:12] == plain[0]["example"][:12]
