"""The multi-hop lookup probe: its lines (experiments/hop_tasks.py), its corpus as the trainer reads it, and its scoring
(experiments/hop_eval.py) for a stack read as a stream and a Transformer read in windows."""
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
import hop_eval  # noqa: E402
from hop_tasks import ASSIGNMENTS, chain_of, example, resolve, write_chain, write_corpus, write_tests  # noqa: E402
from train_event_lm import load_corpus  # noqa: E402


@pytest.mark.parametrize("hops", [1, 2, 3, 4, 8, 16, ASSIGNMENTS])
def test_a_line_takes_its_hops_to_its_answer_and_its_length_does_not_tell_them(hops):
    rng = random.Random(hops)
    for _ in range(50):
        line = example(rng, hops)
        body, query = line.rstrip("\n").rsplit(";?", 1)
        pairs = body.split(";")
        names = [pair.split("=")[0] for pair in pairs]
        assert len(pairs) == ASSIGNMENTS and len(set(names)) == ASSIGNMENTS
        answer, taken = resolve(line)
        assert taken == hops and query.endswith("=" + answer) and line.endswith(answer + "\n")


def test_a_chain_longer_than_the_line_is_refused():
    with pytest.raises(ValueError, match="does not fit"):
        example(random.Random(0), ASSIGNMENTS + 1)


def test_the_corpus_splits_at_line_ends_and_the_trainer_reads_it(tmp_path):
    out = tmp_path / "hops.bin"
    sidecar = write_corpus(out, [1, 2], seed=3, train=5000, validation=1000, test=1000)
    train, validation, test = load_corpus(out)
    assert (len(train), len(validation), len(test)) == (sidecar["train"], sidecar["validation"], sidecar["test"])
    for split in (train, validation, test):
        text = split.tobytes().decode()
        assert text.endswith("\n") and all(resolve(line + "\n")[1] in (1, 2) for line in text.split("\n")[:-1])
    sidecar_path = Path(f"{out}.json")
    broken = dict(json.loads(sidecar_path.read_text()), train=sidecar["train"] + 1)
    sidecar_path.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="sidecar names"):
        load_corpus(out)


def test_answer_places_are_the_last_byte_of_each_scored_line():
    text = example(random.Random(1), 2) + example(random.Random(2), 3) + example(random.Random(3), 1)
    stream, places = hop_eval.answer_places(text.encode(), lead=1)
    lines = text.split("\n")[:-1]
    assert len(places) == 2
    assert [chr(int(stream[p])) for p in places] == [lines[1][-1], lines[2][-1]]
    assert all(chr(int(stream[p - 1])) == "=" for p in places)


def test_the_wilson_interval_holds_the_rate():
    low, high = hop_eval.wilson(30, 100)
    assert low < 0.3 < high and 0 <= low and high <= 1


_STACK = dict(dim=64, hidden=4, depth=2, route="steered", carry=False, resolution=16, router_charge_bits=8, stack=True,
              read_after=(0, 1), read_width=32, read_heads=4, read_window=48, read_mode="race", read_budget=64,
              read_resolution=4, read_address_rate=8.0, granule_after=(0, 1), granule_clusters=1, granule_fan=64,
              granule_cells=96, granule_slope=True, read_delays=4, granule_delays=4, read_cap=128, granule_cap=128)


def test_both_kinds_of_model_are_scored_at_every_answer(tmp_path):
    write_tests(tmp_path, [1, 3], count=24, seed=5)
    torch.manual_seed(0)
    stack = EventLM(EventLMConfig(**_STACK)).eval()
    transformer = TransformerLM(SNNConfig(neuron="ann", attn="transformer", position_embeddings=True),
                                vocab=256, dim=32, depth=2, heads=4, ctx=64).eval()
    for model in (stack, transformer):
        found = hop_eval.score(model, tmp_path / "hops-3.txt", "cpu", window=64, lead=4)
        assert found["answers"] == 20 and 0 <= found["accuracy"] <= 1 and found["answer_bits"] > 0
    # A stack reads the whole file as one stream: scoring it in pieces of any size gives the same scores.
    stream, places = hop_eval.answer_places((tmp_path / "hops-1.txt").read_bytes(), lead=4)
    whole = hop_eval.answer_logits(stack, stream, places, "cpu", 64, piece=4096)
    cut = hop_eval.answer_logits(stack, stream, places, "cpu", 64, piece=37)
    assert torch.equal(whole, cut)


@pytest.mark.parametrize("hops", [1, 3, 16])
def test_a_written_chain_is_the_query_s_chain_and_keeps_the_line_s_assignments(hops):
    line = example(random.Random(3), hops)
    steps = chain_of(line)
    assert len(steps) == hops + 1 and steps[-1].isdigit() and steps[0] == line.rsplit(";?", 1)[1].split("=")[0]
    written = write_chain(line)
    assert written.startswith(line.rsplit(";?", 1)[0] + ";?") and written.endswith("=".join(steps) + "\n")
    assert resolve(written) == resolve(line) == (steps[-1], hops) and write_chain(written) == written


def test_a_corpus_of_written_chains_holds_the_same_lines_with_their_chains(tmp_path):
    plain = write_corpus(tmp_path / "plain.bin", [1, 2, 3, 4], 5, 3000, 500, 500)
    chains = write_corpus(tmp_path / "chains.bin", [1, 2, 3, 4], 5, 3000, 500, 500, written_chains=True)
    a = (tmp_path / "plain.bin").read_text().splitlines()
    b = (tmp_path / "chains.bin").read_text().splitlines()
    assert len(b) > 20 and all(write_chain(x + "\n") == y + "\n" for x, y in zip(a, b))
    assert chains["written_chains"] is True and "written_chains" not in plain
    assert sum(chains[k] for k in ("train", "validation", "test")) == len("".join(y + "\n" for y in b))
