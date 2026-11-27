"""Families of chained tasks for the one-model plan.

A family draws one-line examples with a checkable answer, the last byte of the line, and a chain of single steps
that leads to it, written in the task's own syntax. Every family has one knob, its steps, and a line of 24 items
whatever the steps, so that a line's length does not tell them.

  hops   lookups along a chain of names (hop_tasks.py). The query asks for the digit a name stands for:
             qa=7;mx=qa;zt=mx;...;?zt=7              written: ?zt=mx=qa=7
  offs   lookups that add as they go. A name is bound to a digit or to another name plus a digit, and the query
         asks for the last digit of the sum along its chain: zt = mx + 2, mx = qa + 3, qa = 7, so zt = 12, a 2.
             qa=7;mx=qa3;zt=mx2;...;?zt=2            written: ?zt=mx2=qa5=2
         Each written step is one lookup and one addition of two digits: the name the chain goes on with and the
         sum carried so far. Because every link adds a digit drawn evenly, the answer is even over the ten digits
         whatever the line holds: a guess is right a tenth of the time at every length of two hops or more, where
         the hop family's guess gains with the hops (hop_tasks.py floors).
  sums   the running sum of a channel's readings, read like telemetry. A row holds a channel's readings, one digit
         each, in the order they were taken; the query asks for the last digit of their sum: mx read 7, 7 and 4,
         which add to 18, an 8.
             qa:3152;mx:774;...;+mx=8                written: +mx=7=4=8
         Each written step is the last digit of the sum after one more reading. The step is found by its place in
         the row and not by a name, and the answer is even over the digits from two readings on.

In the other families as in hops, the items beside the query's are chains or rows of one to four steps, so the
corpus of one to four steps shows nothing longer.

Two outputs: a corpus of several families to train on, its lines drawn evenly from the families and from the step
counts (raw bytes with a sidecar of its split sizes, as train_event_lm.py --corpus reads it), and the guess floors
of a family.

usage:
  python experiments/task_families.py corpus --out data/onemodel/chains/three-1-4-written.bin \
      --families hops offs sums --steps 1 2 3 4 --seed 1 --written-chains
  python experiments/task_families.py floors --family offs --steps 1 2 3 4 6 8 12 16 20 --count 4000 --seed 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import hop_tasks
from hop_tasks import NAMES

ITEMS = 24
LONGEST_OTHER = 4


def _lengths(rng: random.Random, steps: int, items: int) -> list[int]:
    """The query's length and, beside it, lengths of one to LONGEST_OTHER until the line holds its items."""
    if not 1 <= steps <= items:
        raise ValueError(f"{steps} steps do not fit in a line of {items} items")
    lengths, left = [steps], items - steps
    while left:
        length = rng.randint(1, min(LONGEST_OTHER, left))
        lengths.append(length)
        left -= length
    return lengths


class Hops:
    """Lookups along a chain of names: hop_tasks.py."""
    name = "hops"
    query = "?"

    @staticmethod
    def example(rng: random.Random, steps: int) -> str:
        return hop_tasks.example(rng, steps)

    @staticmethod
    def chain(line: str) -> list[str]:
        """What is written after the query's '=': the names the query leads through, then the digit."""
        return hop_tasks.chain_of(line)[1:]

    @staticmethod
    def steps(line: str) -> int:
        return len(hop_tasks.chain_of(line)) - 1


class Offs:
    """Lookups that add as they go."""
    name = "offs"
    query = "?"

    @staticmethod
    def example(rng: random.Random, steps: int, items: int = ITEMS) -> str:
        """One line: ``items`` bindings in random order, the query of a chain of ``steps`` of them, its answer."""
        lengths = _lengths(rng, steps, items)
        names = rng.sample(NAMES, items)
        pairs, used, query, answer = [], 0, None, None
        for length in lengths:
            links = names[used:used + length]
            used += length
            value = rng.randrange(10)
            pairs.append((links[0], str(value)))
            for below, above in zip(links, links[1:]):
                added = rng.randrange(10)
                pairs.append((above, f"{below}{added}"))
                value = (value + added) % 10
            if query is None:
                query, answer = links[-1], value
        rng.shuffle(pairs)
        return ";".join(f"{name}={value}" for name, value in pairs) + f";?{query}={answer}\n"

    @staticmethod
    def chain(line: str) -> list[str]:
        """What is written after the query's '=': at every hop the name the chain goes on with and the last digit
        of what was added so far, then the answer (?zt= then mx2, qa5, 2)."""
        body, query = line.rstrip("\n").rsplit(";?", 1)
        bound = dict(pair.split("=", 1) for pair in body.split(";"))
        name, carried, written = query.split("=", 1)[0], 0, []
        while True:
            value = bound[name]
            if value.isdigit():
                written.append(str((int(value) + carried) % 10))
                return written
            name, carried = value[:2], (carried + int(value[2])) % 10
            written.append(f"{name}{carried}")

    @staticmethod
    def steps(line: str) -> int:
        return len(Offs.chain(line))


class Sums:
    """The running sum of a channel's readings."""
    name = "sums"
    query = "+"

    @staticmethod
    def example(rng: random.Random, steps: int, items: int = ITEMS) -> str:
        """One line: rows of ``items`` readings in all, in random order, the query of a row of ``steps`` of them,
        the last digit of its sum."""
        lengths = _lengths(rng, steps, items)
        names = rng.sample(NAMES, len(lengths))
        rows = [(name, "".join(str(rng.randrange(10)) for _ in range(length))) for name, length in zip(names, lengths)]
        query, readings = rows[0]
        rng.shuffle(rows)
        return ";".join(f"{name}:{digits}" for name, digits in rows) + f";+{query}={sum(map(int, readings)) % 10}\n"

    @staticmethod
    def chain(line: str) -> list[str]:
        """What is written after the query's '=': the last digit of the sum after each reading (+mx= then 7, 4, 8)."""
        body, query = line.rstrip("\n").rsplit(";+", 1)
        rows = dict(row.split(":", 1) for row in body.split(";"))
        total, written = 0, []
        for reading in rows[query.split("=", 1)[0]]:
            total = (total + int(reading)) % 10
            written.append(str(total))
        return written

    @staticmethod
    def steps(line: str) -> int:
        return len(Sums.chain(line))


FAMILIES = {family.name: family for family in (Hops, Offs, Sums)}


def _query_at(line: str) -> int:
    """Where the query of a line begins: after its last ';'."""
    return line.rindex(";") + 1


def prompt_of(line: str) -> str:
    """A line up to and with its query's '=': what a model hears before it writes. The same for a line that gives
    the answer alone and for one that writes the chain."""
    return line[:line.index("=", _query_at(line)) + 1]


def written(family: str, line: str) -> str:
    """The line with its query's chain written out before the answer."""
    return prompt_of(line) + "=".join(FAMILIES[family].chain(line)) + "\n"


def answer_of(family: str, line: str) -> str:
    """The answer that a line's own items give its query."""
    return FAMILIES[family].chain(line)[-1]


def guess_floors(family: str, steps: int, count: int = 4000, seed: int = 3) -> dict[str, float]:
    """How often guesses that follow no chain answer a line of ``steps`` steps, as means over ``count`` lines: a
    digit drawn evenly from the digits the line holds; the digit the line holds most often (evenly among the most
    frequent); and the nearest digit: for sums the last reading of the query's row, for the others a digit drawn
    evenly from those that a name is bound to directly. Each guess is taken at its expectation, with no draw."""
    rng = random.Random(seed * 1000 + steps)
    totals = {"a digit of the line": 0.0, "the line's most frequent digit": 0.0, "the nearest digit": 0.0}
    for _ in range(count):
        line = FAMILIES[family].example(rng, steps)
        answer = answer_of(family, line)
        body = line[:_query_at(line)]
        digits = [c for c in body if c.isdigit()]
        totals["a digit of the line"] += digits.count(answer) / len(digits)
        most = max(digits.count(d) for d in set(digits))
        modes = [d for d in set(digits) if digits.count(d) == most]
        totals["the line's most frequent digit"] += (answer in modes) / len(modes)
        if family == "sums":
            name = line[_query_at(line) + 1:].split("=", 1)[0]
            rows = dict(row.split(":", 1) for row in body.rstrip(";").split(";"))
            totals["the nearest digit"] += float(rows[name][-1] == answer)
        else:
            ends = [pair.split("=", 1)[1] for pair in body.rstrip(";").split(";")]
            ends = [value for value in ends if value.isdigit()]
            totals["the nearest digit"] += ends.count(answer) / len(ends)
    return {key: round(total / count, 4) for key, total in totals.items()}


def write_corpus(out: Path, families: list[str], steps: list[int], seed: int, train: int, validation: int, test: int,
                 written_chains: bool = False) -> dict:
    """Lines of the given families and step counts, each drawn evenly, until each split holds at least its bytes;
    split at line ends. With ``written_chains`` every line writes its query's chain out."""
    rng = random.Random(seed)
    sizes, parts, lines_of = {}, [], dict.fromkeys(families, 0)
    for split, want in (("train", train), ("validation", validation), ("test", test)):
        lines, total = [], 0
        while total < want:
            family = rng.choice(families)
            line = FAMILIES[family].example(rng, rng.choice(steps))
            line = (written(family, line) if written_chains else line).encode()
            lines.append(line)
            lines_of[family] += 1
            total += len(line)
        parts.append(b"".join(lines))
        sizes[split] = total
    data = b"".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sidecar = {"format": "raw-bytes-1", "task": "chains", "families": families, "steps": steps, "items": ITEMS,
               "seed": seed, **({"written_chains": True} if written_chains else {}), "lines": lines_of,
               "sha256": hashlib.sha256(data).hexdigest(), **sizes}
    Path(f"{out}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--families", nargs="+", choices=sorted(FAMILIES), required=True)
    corpus.add_argument("--steps", type=int, nargs="+", required=True)
    corpus.add_argument("--seed", type=int, default=1)
    corpus.add_argument("--train-bytes", type=int, default=64_000_000)
    corpus.add_argument("--validation-bytes", type=int, default=2_000_000)
    corpus.add_argument("--test-bytes", type=int, default=2_000_000)
    corpus.add_argument("--written-chains", action="store_true", help="every line writes its query's chain out")
    floors = sub.add_parser("floors")
    floors.add_argument("--family", choices=sorted(FAMILIES), required=True)
    floors.add_argument("--steps", type=int, nargs="+", required=True)
    floors.add_argument("--count", type=int, default=4000)
    floors.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    if args.what == "corpus":
        print(json.dumps(write_corpus(args.out, args.families, args.steps, args.seed, args.train_bytes,
                                      args.validation_bytes, args.test_bytes, args.written_chains)))
    else:
        print(json.dumps({str(s): guess_floors(args.family, s, args.count, args.seed) for s in args.steps}, indent=1))


if __name__ == "__main__":
    main()
