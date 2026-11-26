"""Multi-hop variable lookups as byte streams: a probe of how many steps a model chains without writing them out.

Every example is one line: assignments in random order, then a query, then its answer.

    qa=7;mx=qa;fb=3;zt=mx;...;?zt=7

An assignment binds a two-letter name to a digit or to another name. The query asks for the digit that a name stands
for, which takes following its chain of names: here zt -> mx -> qa -> 7, three hops. The other assignments form
chains of their own, so every example holds the same number of assignments whatever its hops, and its length does
not tell them. The answer is the byte after the query's '='. A model that chains h hops gets it right; one that only
knows the answer is a digit gets a tenth.

Two outputs: a corpus to train on (raw bytes with a sidecar of its split sizes, as train_event_lm.py --corpus reads
it), and test files of one hop count each. With --written-chains every line of the corpus writes its query's chain out
before the answer, the same lines otherwise (the same seed draws the same assignments):

    qa=7;mx=qa;fb=3;zt=mx;...;?zt=mx=qa=7

so that each byte after the query's '=' is a lookup of one hop: a chain of thought in the task's own syntax.

usage:
  python experiments/hop_tasks.py corpus --out data/onemodel/hops/hops-1-4.bin --hops 1 2 3 4 --seed 1
  python experiments/hop_tasks.py corpus --out data/onemodel/hops/hops-1-4-written.bin --hops 1 2 3 4 --seed 1 \
      --written-chains
  python experiments/hop_tasks.py tests --out data/onemodel/hops/tests --hops 1 2 3 4 6 8 12 16 --count 1000 --seed 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from pathlib import Path

NAMES = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase]
ASSIGNMENTS = 24
LONGEST_DISTRACTOR = 4


def example(rng: random.Random, hops: int, assignments: int = ASSIGNMENTS) -> str:
    """One line: ``assignments`` bindings in random order, the query of a chain of ``hops`` of them, its answer."""
    if not 1 <= hops <= assignments:
        raise ValueError(f"a chain of {hops} hops does not fit in {assignments} assignments")
    names = rng.sample(NAMES, 2 * assignments)
    pairs, used = [], 0

    def chain(length):
        nonlocal used
        links = names[used:used + length]
        used += length
        digit = str(rng.randrange(10))
        pairs.extend((links[i], links[i - 1]) for i in range(1, length))
        pairs.append((links[0], digit))
        return links[-1], digit

    query, answer = chain(hops)
    left = assignments - hops
    while left:
        length = rng.randint(1, min(LONGEST_DISTRACTOR, left))
        chain(length)
        left -= length
    rng.shuffle(pairs)
    return ";".join(f"{name}={value}" for name, value in pairs) + f";?{query}={answer}\n"


def chain_of(line: str) -> list[str]:
    """The query's chain as a line's assignments give it: the query's name, the names it leads through, the digit."""
    body, query = line.rstrip("\n").rsplit(";?", 1)
    name = query.split("=", 1)[0]
    bound = dict(pair.split("=", 1) for pair in body.split(";"))
    steps = [name]
    while not name.isdigit():
        name = bound[name]
        steps.append(name)
    return steps


def resolve(line: str) -> tuple[str, int]:
    """The answer that a line's assignments give its query, and the hops it took."""
    steps = chain_of(line)
    return steps[-1], len(steps) - 1


def write_chain(line: str) -> str:
    """The line with its query's chain written out before the answer: ?zt=7 becomes ?zt=mx=qa=7."""
    body = line.rstrip("\n").rsplit(";?", 1)[0]
    return f"{body};?{'='.join(chain_of(line))}\n"


def write_corpus(out: Path, hops: list[int], seed: int, train: int, validation: int, test: int,
                 written_chains: bool = False):
    """Lines of the given hop counts, drawn evenly, until each split holds at least its bytes; split at line ends.
    With ``written_chains`` every line writes its query's chain out (write_chain)."""
    rng = random.Random(seed)
    sizes, parts = {}, []
    for split, want in (("train", train), ("validation", validation), ("test", test)):
        lines, total = [], 0
        while total < want:
            line = example(rng, rng.choice(hops))
            line = (write_chain(line) if written_chains else line).encode()
            lines.append(line)
            total += len(line)
        parts.append(b"".join(lines))
        sizes[split] = total
    data = b"".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    sidecar = {"format": "raw-bytes-1", "task": "hops", "hops": hops, "assignments": ASSIGNMENTS, "seed": seed,
               **({"written_chains": True} if written_chains else {}), "sha256": hashlib.sha256(data).hexdigest(),
               **sizes}
    Path(f"{out}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return sidecar


def write_tests(out: Path, hops: list[int], count: int, seed: int):
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for h in hops:
        rng = random.Random(seed * 1000 + h)
        text = "".join(example(rng, h) for _ in range(count))
        path = out / f"hops-{h}.txt"
        path.write_text(text)
        written[h] = hashlib.sha256(text.encode()).hexdigest()
    (out / "tests.json").write_text(json.dumps({"assignments": ASSIGNMENTS, "count": count, "seed": seed,
                                                "sha256": {str(h): s for h, s in written.items()}}, indent=1) + "\n")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--out", type=Path, required=True)
    corpus.add_argument("--hops", type=int, nargs="+", required=True)
    corpus.add_argument("--seed", type=int, default=1)
    corpus.add_argument("--train-bytes", type=int, default=64_000_000)
    corpus.add_argument("--validation-bytes", type=int, default=2_000_000)
    corpus.add_argument("--test-bytes", type=int, default=2_000_000)
    corpus.add_argument("--written-chains", action="store_true", help="every line writes its query's chain out")
    tests = sub.add_parser("tests")
    tests.add_argument("--out", type=Path, required=True)
    tests.add_argument("--hops", type=int, nargs="+", required=True)
    tests.add_argument("--count", type=int, default=1000)
    tests.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    if args.what == "corpus":
        print(json.dumps(write_corpus(args.out, args.hops, args.seed, args.train_bytes, args.validation_bytes,
                                      args.test_bytes, args.written_chains)))
    else:
        print(json.dumps({str(h): s for h, s in write_tests(args.out, args.hops, args.count, args.seed).items()}))


if __name__ == "__main__":
    main()
