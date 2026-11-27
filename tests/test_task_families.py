"""The task families of the one-model plan (experiments/task_families.py): their lines, chains, floors and corpus."""
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import hop_tasks  # noqa: E402
import task_families as families  # noqa: E402


@pytest.mark.parametrize("family", sorted(families.FAMILIES))
@pytest.mark.parametrize("steps", [1, 2, 4, 9, 24])
def test_a_line_holds_its_items_and_a_chain_of_the_steps_asked(family, steps):
    rng = random.Random(steps)
    line = families.FAMILIES[family].example(rng, steps)
    chain = families.FAMILIES[family].chain(line)
    assert len(chain) == steps == families.FAMILIES[family].steps(line)
    assert line.endswith(chain[-1] + "\n") and chain[-1].isdigit() and len(chain[-1]) == 1
    written = families.written(family, line)
    assert families.prompt_of(written) == families.prompt_of(line) == line[:-2]
    assert written == line[:-2] + "=".join(chain) + "\n"
    assert families.FAMILIES[family].chain(written[:len(line) - 2] + chain[-1] + "\n") == chain
    body = line[:line.rindex(";")]
    if family == "sums":
        assert sum(len(row.split(":")[1]) for row in body.split(";")) == families.ITEMS
    else:
        assert len(body.split(";")) == families.ITEMS


def test_a_line_longer_than_its_items_is_refused():
    for family in families.FAMILIES.values():
        with pytest.raises(ValueError):
            family.example(random.Random(0), families.ITEMS + 1)


def test_the_hop_family_is_the_hop_tasks():
    ours, theirs = random.Random(4), random.Random(4)
    for steps in (1, 3, 8):
        line = hop_tasks.example(theirs, steps)
        assert families.Hops.example(ours, steps) == line
        assert families.written("hops", line) == hop_tasks.write_chain(line)


def test_offs_add_along_the_chain():
    line = "qa=7;mx=qa3;fb=4;zt=mx2;?zt=2\n"
    assert families.Offs.chain(line) == ["mx2", "qa5", "2"]
    assert families.written("offs", line) == "qa=7;mx=qa3;fb=4;zt=mx2;?zt=mx2=qa5=2\n"
    assert families.Offs.chain("qa=7;mx=qa3;fb=4;zt=mx2;?fb=4\n") == ["4"]
    assert families.Offs.chain("qa=7;mx=qa3;fb=4;zt=mx9;?zt=9\n") == ["mx9", "qa2", "9"]


def test_sums_carry_the_last_digit_reading_by_reading():
    line = "qa:3152;mx:774;fb:4;+mx=8\n"
    assert families.Sums.chain(line) == ["7", "4", "8"]
    assert families.written("sums", line) == "qa:3152;mx:774;fb:4;+mx=7=4=8\n"
    assert families.Sums.chain("qa:3152;mx:774;fb:4;+fb=4\n") == ["4"]
    assert families.Sums.chain("qa:3152;mx:774;fb:4;+qa=1\n") == ["3", "4", "9", "1"]


@pytest.mark.parametrize("family", ["offs", "sums"])
def test_a_guess_is_right_a_tenth_of_the_time_from_two_steps_on(family):
    for steps in (2, 8, 20):
        floors = families.guess_floors(family, steps, count=1500, seed=5)
        assert all(0.07 < share < 0.13 for share in floors.values()), (steps, floors)


def test_the_answers_are_even_over_the_digits():
    rng = random.Random(8)
    for family in ("offs", "sums"):
        counts = [0] * 10
        for _ in range(4000):
            counts[int(families.answer_of(family, families.FAMILIES[family].example(rng, 6)))] += 1
        assert min(counts) > 320 and max(counts) < 480, (family, counts)


def test_a_corpus_draws_its_families_evenly_and_writes_their_chains(tmp_path):
    out = tmp_path / "three.bin"
    sidecar = families.write_corpus(out, ["hops", "offs", "sums"], [1, 2, 3, 4], 1, 60_000, 6_000, 6_000, True)
    data = out.read_bytes()
    assert json.loads(Path(f"{out}.json").read_text()) == sidecar and sidecar["written_chains"] is True
    assert len(data) == sidecar["train"] + sidecar["validation"] + sidecar["test"] and sidecar["train"] >= 60_000
    lines = data.decode().splitlines(keepends=True)
    assert sum(sidecar["lines"].values()) == len(lines) and min(sidecar["lines"].values()) > len(lines) / 4
    for line in lines[:300]:
        items = line[:line.rindex(";")].split(";")  # a line of offs whose chains are all one hop long reads as hops
        family = "sums" if ";+" in line else "offs" if any(len(i.split("=")[1]) == 3 for i in items) else "hops"
        prompt = families.prompt_of(line)
        steps = line[len(prompt):-1].split("=")
        assert families.FAMILIES[family].chain(prompt + steps[-1] + "\n") == steps and 1 <= len(steps) <= 4
    again = families.write_corpus(tmp_path / "again.bin", ["hops", "offs", "sums"], [1, 2, 3, 4], 1, 60_000, 6_000,
                                  6_000, True)
    assert again["sha256"] == sidecar["sha256"]
