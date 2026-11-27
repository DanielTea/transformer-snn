"""Recall by distance: one-hop lookups whose assignment stands a set number of bytes back (run from the root of a
copy of the repository's code):

    python experiments/recall_over_distance.py OUT.json NAME=CHECKPOINT [NAME=CHECKPOINT ...] [--tasks 200] [--seed 0]

Lines of 24 and of 36 assignments as experiments/hop_tasks.py draws them, each with a query of one hop. In every line
the query's assignment is moved to a place drawn among those whose distance back from the end of the prompt (the
query's '=', where the answer is written) lies in a band of 25 bytes: up to 125-150 in lines of 24 assignments, whose
prompts are 131 to 141 bytes long, and up to 200-225 in lines of 36 (199 to 208 bytes). The distance runs from the
first byte of the assignment, as in experiments/where_chains_break.py. A line in which no place falls in the band is
drawn again. The other assignments keep their order, so a line differs from a line of the corpus only in its length
and in where its query's assignment stands. The same tasks for every model: --tasks lines a band and line length,
drawn from --seed. Every model answers greedily with room for 16 bytes, each answer in a stream of its own from an
empty state (experiments/practice_loop.py attempts), and an answer is right if the last byte before the newline is
the query's digit (practice_loop.grade). Reported per model, line length and band: right answers, the accuracy with
its 95% Wilson interval, the share of answers that are one digit and a newline, the mean bytes."""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, "experiments")
import hop_tasks  # noqa: E402
import practice_loop as P  # noqa: E402

BANDS = {24: [(lo, lo + 25) for lo in range(0, 150, 25)], 36: [(lo, lo + 25) for lo in range(0, 225, 25)]}


def wilson(right: int, n: int, z: float = 1.959964) -> list[float]:
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(centre - half, 4), round(centre + half, 4)]


def placed(rng: random.Random, assignments: int, lo: int, hi: int) -> tuple[P.Task, int]:
    """A line of ``assignments`` with a query of one hop whose assignment stands lo <= distance < hi bytes back."""
    while True:
        line = hop_tasks.example(rng, 1, assignments)
        body, query = line.rstrip("\n").rsplit(";?", 1)
        name, digit = query.split("=", 1)
        pairs = body.split(";")
        own = f"{name}={digit}"
        rest = [p for p in pairs if p != own]
        assert len(rest) == len(pairs) - 1
        prompt_length = len(body) + 2 + len(name) + 1  # the assignments, ';?', the query's name, '='
        starts, at = [], 0
        for j in range(len(rest) + 1):  # the place before rest[j]; the last place is after every other assignment
            starts.append(at)
            if j < len(rest):
                at += len(rest[j]) + 1
        fits = [j for j, start in enumerate(starts) if lo <= prompt_length - start < hi]
        if not fits:
            continue
        j = rng.choice(fits)
        moved = rest[:j] + [own] + rest[j:]
        prompt = ";".join(moved) + f";?{name}="
        assert len(prompt) == prompt_length and prompt[starts[j]:starts[j] + len(own) + 1] == own + ";"
        distance = prompt_length - starts[j]  # from the assignment's first byte
        assert lo <= distance < hi, (distance, lo, hi)
        task = P.Task(prompt.encode(), ord(digit), 1, "hops")
        assert P.verify(task)
        return task, distance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--tasks", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = {}
    for assignments, bands in BANDS.items():
        for k, (lo, hi) in enumerate(bands):
            rng = random.Random(args.seed * 1_000_003 + assignments * 1000 + k)
            tasks[(assignments, lo, hi)] = [placed(rng, assignments, lo, hi) for _ in range(args.tasks)]
    found = {"tasks a band": args.tasks, "seed": args.seed, "models": {}}
    for spec in args.models:
        name, path = spec.split("=", 1)
        model, _ = P.load_checkpoint(Path(path), device)
        model.eval()
        cells = {}
        for (assignments, lo, hi), drawn in tasks.items():
            batch = [task for task, _ in drawn]
            with torch.no_grad():
                written = P.attempts(model, batch, 1, 16, 1e-6, torch.Generator().manual_seed(args.seed), device)
            right = sum(P.grade(t, w) == 1.0 for t, w in zip(batch, written))
            form = sum(len(w) == 2 and chr(w[0]).isdigit() and w[1] == P.NEWLINE for w in written)
            cells.setdefault(str(assignments), {})[f"{lo}-{hi}"] = {
                "right": right, "of": len(batch), "accuracy": round(right / len(batch), 4),
                "interval_95": wilson(right, len(batch)), "one digit and a newline": round(form / len(batch), 4),
                "mean_bytes": round(sum(len(w) for w in written) / len(batch), 2),
                "mean distance": round(sum(d for _, d in drawn) / len(drawn), 1)}
        found["models"][name] = cells
        print(name, {a: {b: c["accuracy"] for b, c in v.items()} for a, v in cells.items()}, flush=True)
        Path(args.out).write_text(json.dumps(found, indent=1) + "\n")


if __name__ == "__main__":
    main()
