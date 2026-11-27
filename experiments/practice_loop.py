"""Practice with feedback: a model learns from its own attempts at tasks kept at the edge of what it can do.

The loop of the drawing in the one-model plan, with programs where the drawing has helper models (the plan admits no
other model as a teacher):
  generator  draws tasks of a family at a difficulty (here: lookups along a chain of ``hops`` names, experiments/
             hop_tasks.py; the line is the prompt up to the query's '=');
  verifier   keeps a task only if its reference answer follows from its own assignments and the task is not answered
             by the query's own name;
  model      writes ATTEMPTS continuations of every task as one stream each, a byte at a time, up to a newline or
             MAX_BYTES bytes, at a temperature;
  grader     takes the last byte before the newline as the answer and gives a reward of 1 if it is the right digit;
  solve rate per difficulty; the improver moves the difficulty by a staircase so that the solve rate stays between
             a quarter and a half (above a half it adds a hop, below a quarter it takes one away);
  learning   a policy-gradient step on the attempts of every task whose attempts disagree (the group's rewards minus
             their mean, over their spread), on the bytes the model wrote, as GRPO does; tasks that all attempts
             solve or all fail teach nothing and are left out.
The model may write anything before its answer. Between the query's '=' and the answer digit it can write the names of
the chain (?zt=mx=qa=7), each one lookup that a shallow model can make: a chain of thought in the task's own syntax,
which no program writes for it. The share of attempts that write at least one name is reported every round.

A guess pays more the longer the chain: a line holds 24 assignments whatever its hops, so a long chain leaves few
other digits, and a guess among the digits that a name points to is right 0.20 of the time at 4 hops, 0.24 at 8, 0.37
at 16 and more than half the time from 20 hops. A staircase that climbs there rewards the guess as it rewards the
chain. --longest-hops caps it where a guess stays under the staircase's lower mark (8 hops). --evaluate-every scores
greedy answers along the way, so that what practice does to the written chains is seen before the last round.
--dips weighs what an attempt without reward teaches (policy_loss): at 0 the model learns only from the attempts
that were rewarded, as a synapse that is strengthened by a burst of dopamine and barely weakened by its pause.

Practice alone holds nothing in place. Once a length is solved its rounds teach almost nothing, the optimizer's steps
stay as large as they were, and after some hundred rounds a model can lose what it had, the stop of a chain first
(practice_within_reach.json). --rehearse adds to every round what the written corpus taught: fresh tasks of the
corpus's lengths with their chains written as the corpus writes them, learned as the corpus was (rehearsal_loss).

usage: python experiments/practice_loop.py --checkpoint run/model.pt.averaged --out practice/ --rounds 200 \\
    [--device mps] [--longest-hops 8] [--evaluate-every 50 --evaluate-hops 4 6 8] [--dips 0] [--rehearse 16]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import resolve_device  # noqa: E402
from hop_tasks import chain_of, example, resolve  # noqa: E402
from snnformer.event_lm import EventLM  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402

NEWLINE = ord("\n")


def scores(model, output):
    """A model's output as scores over the 256 bytes: a stack's spike evidence, a Transformer's logits."""
    if isinstance(model, EventLM):
        return scoring_logits(output, True, model.cfg.resolution)
    return output


@dataclass
class Task:
    prompt: bytes      # the line up to and with the query's '='
    answer: int        # the answer digit's byte
    hops: int


def generate(rng: random.Random, hops: int) -> Task:
    line = example(rng, hops)
    cut = line.rindex("=") + 1
    return Task(line[:cut].encode(), ord(line[cut]), hops)


def verify(task: Task) -> bool:
    """The reference answer follows from the task's own assignments, and the query is not a digit itself."""
    text = task.prompt.decode() + chr(task.answer) + "\n"
    answer, hops = resolve(text)
    query = task.prompt.decode().rsplit(";?", 1)[1][:-1]
    return answer == chr(task.answer) and hops == task.hops and not query.isdigit()


def grade(task: Task, written: bytes) -> float:
    """1 if the last byte before the newline is the right digit; an attempt without a newline gets nothing."""
    if NEWLINE not in written:
        return 0.0
    body = written[:written.index(NEWLINE)]
    return float(len(body) > 0 and body[-1] == task.answer)


def staircase(hops: int, solve_rate: float, low: float = 0.25, high: float = 0.5, longest: int = 24) -> int:
    if solve_rate > high:
        return min(hops + 1, longest)
    if solve_rate < low:
        return max(hops - 1, 1)
    return hops


@torch.no_grad()
def attempts(model, tasks: list[Task], count: int, max_bytes: int, temperature: float, generator, device):
    """``count`` continuations of every task, written in parallel streams a byte at a time."""
    prompts = [task.prompt for task in tasks for _ in range(count)]
    width = max(len(p) for p in prompts)
    # Every stream hears its own prompt; shorter prompts are led by newlines, which end a line of the corpus.
    heard = torch.tensor([[NEWLINE] * (width - len(p)) + list(p) for p in prompts], dtype=torch.long, device=device)
    written = [bytearray() for _ in prompts]
    done = [False] * len(prompts)
    stack = isinstance(model, EventLM)
    # A stack carries its state from byte to byte; a Transformer reads its whole window again for every byte.
    counts, state = model.stream(heard, None) if stack else (model(heard), None)
    for _ in range(max_bytes):
        logits = scores(model, counts)[:, -1].cpu().double()  # to the CPU first: Apple's GPU has no float64
        chances = (logits / temperature).softmax(-1)
        drawn = torch.multinomial(chances, 1, generator=generator).squeeze(-1)
        for row, byte in enumerate(drawn.tolist()):
            if not done[row]:
                written[row].append(byte)
                done[row] = byte == NEWLINE
        if all(done):
            break
        if stack:
            counts, state = model.stream(drawn[:, None].to(device), state)
        else:
            heard = torch.cat((heard, drawn[:, None].to(device)), 1)
            counts = model(heard)
    return [bytes(w) for w in written]


def policy_loss(model, tasks: list[Task], written: list[bytes], rewards: torch.Tensor, count: int, device,
                dips: float = 1.0):
    """The GRPO loss on the bytes written for the tasks whose attempts disagree, or None if none do. An attempt
    teaches by how far its reward came out above or below the mean of its task's attempts, in their standard
    deviations: above it, its bytes become likelier (a burst); below it, less likely (a dip). ``dips`` weighs the
    dips: 1 is GRPO, and at 0 only the attempts that were rewarded teach, an unexpected reward the most. A dip
    lowers the likelihood of bytes the model itself drew, which has no floor; a burst raises one that ends at 1."""
    groups = rewards.view(len(tasks), count)
    spread = groups.std(1, unbiased=False)
    useful = spread > 0
    if not useful.any():
        return None
    advantages = ((groups - groups.mean(1, keepdim=True)) / (spread[:, None] + 1e-6))[useful].flatten()
    if dips != 1.0:
        advantages = torch.where(advantages < 0, advantages * dips, advantages)
    rows = [index for index, keep in enumerate(useful.repeat_interleave(count).tolist()) if keep]
    per_attempt = written_likelihood(model, [tasks[i // count].prompt for i in rows], [written[i] for i in rows],
                                     device)
    return -(advantages.to(device) * per_attempt).mean()


def written_likelihood(model, prompts: list[bytes], written: list[bytes], device):
    """For every row, the mean log-likelihood a byte of what was written after its prompt, the row read as one
    window (shorter rows are led by newlines, as the attempts were)."""
    texts = [prompt + wrote for prompt, wrote in zip(prompts, written)]
    width = max(len(t) for t in texts)
    tokens = torch.tensor([[NEWLINE] * (width - len(t)) + list(t) for t in texts], dtype=torch.long, device=device)
    mask = torch.zeros(len(texts), width - 1, device=device)
    for row, wrote in enumerate(written):
        mask[row, width - 1 - len(wrote):] = 1.0
    logits = scores(model, model(tokens[:, :-1])).float()
    logp = -F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none").view(len(texts), -1)
    return (logp * mask).sum(1) / mask.sum(1).clamp_min(1)


def chain_written(task: Task) -> bytes:
    """What the written corpus writes after a task's prompt: the names its query leads through, the digit and a
    newline (?zt= then mx=qa=7; one hop: 7)."""
    steps = chain_of(task.prompt.decode() + chr(task.answer) + "\n")
    return "=".join(steps[1:]).encode() + b"\n"


def rehearsal_loss(model, tasks: list[Task], device):
    """What the written corpus taught, asked again: the mean cost a byte of writing every task's chain as the corpus
    writes it. No attempt and no reward: the lesson is the one the model was trained on before it practised."""
    return -written_likelihood(model, [task.prompt for task in tasks], [chain_written(task) for task in tasks],
                               device).mean()


def wrote_the_chain(task: Task, written: bytes) -> bool:
    """The answer writes the query's whole chain and nothing else: ?zt= then mx=qa=7 and a newline (one hop: 7)."""
    return written == chain_written(task)


@torch.no_grad()
def evaluate(model, hops_list, count: int, max_bytes: int, seed: int, device) -> dict:
    """Greedy answers with room to think: for every hop count, ``count`` tasks (drawn from ``seed``), the model
    writes its most likely bytes up to a newline; the accuracy of the last byte, the share of answers that write at
    least one name, the share that write the query's whole chain right (wrote_the_chain), and the mean bytes."""
    found = {}
    for hops in hops_list:
        rng = random.Random(seed * 1000 + hops)
        tasks = [generate(rng, hops) for _ in range(count)]
        written = attempts(model, tasks, 1, max_bytes, 1e-6, torch.Generator().manual_seed(seed), device)
        chains = sum(wrote_the_chain(t, w) for t, w in zip(tasks, written))
        found[str(hops)] = {"accuracy": round(sum(grade(t, w) for t, w in zip(tasks, written)) / count, 4),
                            "wrote_names": round(sum(map(wrote_a_name, written)) / count, 4),
                            "wrote_the_chain": round(chains / count, 4),
                            "mean_bytes": round(sum(len(w) for w in written) / count, 2)}
    return found


def wrote_a_name(written: bytes) -> bool:
    body = written.split(b"\n", 1)[0]
    return any(chr(b).isalpha() for b in body)


def keep_evaluation(model, checkpoint, out: Path, round_: int, hops: int, hops_list, count: int, max_bytes: int,
                    seed: int, device) -> dict:
    """Score the model's greedy answers after ``round_`` rounds of practice (evaluate, on the tasks that the
    evaluation of a finished run draws from the same seed), append the row to OUT/evaluations.jsonl and keep the
    scored model as OUT/model-rROUND.pt. The evaluation draws from generators of its own, so the practice that
    follows is the practice without it."""
    was_training = model.training
    model.eval()
    row = {"round": round_, "hops": hops, "by_hops": evaluate(model, hops_list, count, max_bytes, seed, device)}
    model.train(was_training)
    with (out / "evaluations.jsonl").open("a") as log:
        log.write(json.dumps(row) + "\n")
    torch.save({**checkpoint, "state_dict": model.state_dict(), "practice_rounds": round_, "practice_hops": hops},
               out / f"model-r{round_:05d}.pt")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--tasks", type=int, default=16, help="tasks a round")
    parser.add_argument("--attempts", type=int, default=8, help="attempts at every task")
    parser.add_argument("--max-bytes", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--start-hops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--evaluate", type=int, nargs="*",
                        help="only score greedy answers with room to think at these hop counts, into OUT/evaluate.json")
    parser.add_argument("--evaluate-tasks", type=int, default=200)
    parser.add_argument("--longest-hops", type=int, default=24,
                        help="the staircase never asks for more hops than this (24 assignments a line; from 9 hops a "
                             "guess among the line's digits is right more than a quarter of the time)")
    parser.add_argument("--evaluate-every", type=int, default=0, metavar="ROUNDS",
                        help="while practising, score greedy answers before the first round and after every ROUNDS "
                             "rounds at --evaluate-hops (OUT/evaluations.jsonl) and keep each scored model")
    parser.add_argument("--evaluate-hops", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--dips", type=float, default=1.0,
                        help="weight of the attempts that did worse than their task's mean (1: GRPO; 0: only "
                             "rewarded attempts teach)")
    parser.add_argument("--rehearse", type=int, default=0, metavar="TASKS",
                        help="add to every round's step the written chains of TASKS fresh tasks of --rehearse-hops "
                             "(0: practice alone)")
    parser.add_argument("--rehearse-hops", type=int, nargs="+", default=[1, 2, 3, 4],
                        help="the lengths rehearsed, drawn evenly: those of the written corpus")
    parser.add_argument("--rehearse-weight", type=float, default=1.0,
                        help="weight of the rehearsal's loss beside the practice's")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if args.evaluate:
        args.out.mkdir(parents=True, exist_ok=True)
        found = evaluate(model, args.evaluate, args.evaluate_tasks, args.max_bytes, args.seed + 7, device)
        (args.out / "evaluate.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), "by_hops": found},
                                                           indent=1) + "\n")
        print(json.dumps(found), flush=True)
        return
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng, generator = random.Random(args.seed), torch.Generator().manual_seed(args.seed)
    rehearsing = random.Random(args.seed + 13)  # rehearsal draws its own tasks: the practice's stay as they were
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "rounds.jsonl").open("a")
    hops = args.start_hops

    def scored(round_):
        if args.evaluate_every and round_ % args.evaluate_every == 0:
            row = keep_evaluation(model, checkpoint, args.out, round_, hops, args.evaluate_hops, args.evaluate_tasks,
                                  args.max_bytes, args.seed + 7, device)
            print(json.dumps(row), flush=True)

    scored(0)
    for round_ in range(1, args.rounds + 1):
        tasks = [task for task in (generate(rng, hops) for _ in range(args.tasks)) if verify(task)]
        model.eval()
        written = attempts(model, tasks, args.attempts, args.max_bytes, args.temperature, generator, device)
        rewards = torch.tensor([grade(tasks[i // args.attempts], w) for i, w in enumerate(written)])
        solve_rate = float(rewards.mean())
        model.train()
        loss = policy_loss(model, tasks, written, rewards, args.attempts, device, args.dips)
        rehearsal = None
        if args.rehearse:
            again = [generate(rehearsing, rehearsing.choice(args.rehearse_hops)) for _ in range(args.rehearse)]
            rehearsal = rehearsal_loss(model, again, device)
        step = [part for part in (loss, None if rehearsal is None else args.rehearse_weight * rehearsal)
                if part is not None]
        if step:
            optimizer.zero_grad(set_to_none=True)
            (step[0] if len(step) == 1 else step[0] + step[1]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if hasattr(model, "synaptic_scaling"):
                model.synaptic_scaling()
        row = {"round": round_, "hops": hops, "tasks": len(tasks), "solve_rate": round(solve_rate, 4),
               "wrote_names": round(sum(map(wrote_a_name, written)) / len(written), 4),
               "wrote_the_chain": round(sum(wrote_the_chain(tasks[i // args.attempts], w)
                                            for i, w in enumerate(written)) / len(written), 4),
               "mean_bytes": round(sum(len(w) for w in written) / len(written), 2),
               "loss": None if loss is None else round(float(loss.detach()), 5),
               **({} if rehearsal is None else {"rehearsal": round(float(rehearsal.detach()), 5)}),
               "example": (tasks[0].prompt[-12:] + written[0]).decode("latin-1")}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        hops = staircase(hops, solve_rate, longest=args.longest_hops)
        scored(round_)
        if round_ % 50 == 0 or round_ == args.rounds:
            torch.save({**checkpoint, "state_dict": model.state_dict(), "practice_rounds": round_,
                        "practice_hops": hops}, args.out / "model.pt")
    log.close()


if __name__ == "__main__":
    main()
