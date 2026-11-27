"""What a practised model of several families writes (run from the root of a copy of the repository's code):

    python experiments/forms_families.py OUT.jsonl NAME=CHECKPOINT [NAME=CHECKPOINT ...] -- FAMILY:STEPS [...]

Greedy answers to the tasks of experiments/practice_loop.py's evaluation (seed 0, 200 tasks a cell) for each family
and step count, counted by form: the chain as the corpus writes it, an answer that runs to the end of the room (64
bytes) without a newline, a right answer with the chain otherwise, a wrong answer. Beside the forms: answers that
write a name twice or an empty step (==), how many steps each answer writes right before its first wrong one, and
the last byte of the answers. One JSON object a line is appended to OUT.jsonl for every model, family and step
count, with four example answers beside the chains as the corpus writes them. Used for practice_several_families.json
(results/event_lm/one_model/), on the server's CPU."""
import json, random, sys
from collections import Counter
from pathlib import Path
import torch
sys.path.insert(0, "experiments")
import practice_loop as P
out = sys.argv[1]
split = sys.argv.index("--")
models = dict(a.split("=", 1) for a in sys.argv[2:split])
cells = [(c.split(":")[0], int(c.split(":")[1])) for c in sys.argv[split + 1:]]
torch.set_num_threads(4)
with open(out, "a") as sink:
    for name, path in models.items():
        model, _ = P.load_checkpoint(Path(path), torch.device("cpu"))
        model.eval()
        for family, steps in cells:
            rng = random.Random(0 * 1000 + steps)
            tasks = [P.generate(rng, steps, family) for _ in range(200)]
            with torch.no_grad():
                written = P.attempts(model, tasks, 1, 64, 1e-6, torch.Generator().manual_seed(0), torch.device("cpu"))
            kinds, marks, examples = Counter(), Counter(), []
            right_steps, last_byte = Counter(), Counter()
            for t, w in zip(tasks, written):
                right = P.chain_written(t).decode().rstrip("\n")
                text = w.decode("latin-1")
                body = text.split("\n", 1)[0]
                if body == right and "\n" in text:
                    kinds["the chain as the corpus writes it"] += 1
                elif "\n" not in text:
                    kinds["ran to the end of the room (64 bytes) without a newline"] += 1
                elif P.grade(t, w) == 1.0:
                    kinds["right answer, the chain otherwise"] += 1
                else:
                    kinds["wrong answer"] += 1
                steps_written = body.lstrip("?").split("=")
                names = [s[:2] for s in steps_written if len(s) >= 2 and s[:2].isalpha()]
                if len(names) != len(set(names)):
                    marks["a name written twice"] += 1
                if "==" in body:
                    marks["an empty step (==)"] += 1
                want = right.split("=")
                got = body.lstrip("?").split("=")
                k = 0
                while k < min(len(want), len(got)) - 1 and got[k] == want[k]:
                    k += 1
                right_steps[k] += 1
                last_byte[body[-1:] if "\n" in text else "(no newline)"] += 1
                if len(examples) < 4:
                    examples.append({"right": right, "written": body if "\n" in text else body + " ..."})
            row = {"model": name, "family": family, "steps": steps, "forms": dict(kinds), "marks": dict(marks),
                   "steps written right before the first wrong one": dict(sorted(right_steps.items())),
                   "the last byte": dict(last_byte.most_common()),
                   "examples": examples}
            sink.write(json.dumps(row) + "\n"); sink.flush()
            print(json.dumps(row)[:300], flush=True)
