"""Where and how written hop chains go wrong, for practised models (run from the repository's root):

    python experiments/where_chains_break.py OUT.json NAME=CHECKPOINT [NAME=CHECKPOINT ...] -- HOPS [HOPS ...]

For every model and hop count: the tasks of the evaluation after practice (seed 0, 200 tasks, greedy, room for 64
bytes), each answer compared with the right chain step by step (question marks dropped, so a chain written in
r8-tf-s1's form is compared by its names). Per answer: the step of the first error (0 = the first name after the
query) and its kind: a digit written too early, a name written where the digit belongs, a wrong digit at the end,
a skipped step (the name the right one points to), a repeated name, a step backwards (the name that points to the
previous one), another name of the line, a name not in the line. Per step written before or at the first error:
how far back in the stream (bytes from the place it is written) the assignment that it needs stands in the line,
and whether it was written right, so that the error rate can be read against that distance."""
import json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import torch
sys.path.insert(0, "experiments")
import practice_loop as P

out = Path(sys.argv[1])
split = sys.argv.index("--")
models = dict(arg.split("=", 1) for arg in sys.argv[2:split])
hops_list = [int(h) for h in sys.argv[split + 1:]]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(int(__import__("os").environ.get("THREADS", "4")))
BINS = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 250), (250, 400)]
found = {}
for name, path in models.items():
    model, _ = P.load_checkpoint(Path(path), device)
    model.eval()
    found[name] = {}
    for hops in hops_list:
        rng = random.Random(0 * 1000 + hops)
        tasks = [P.generate(rng, hops, "hops") for _ in range(200)]
        with torch.no_grad():
            written = P.attempts(model, tasks, 1, 64, 1e-6, torch.Generator().manual_seed(0), device)
        kinds, first_steps = Counter(), Counter()
        by_distance = defaultdict(lambda: [0, 0])  # bin -> [steps written, steps wrong]
        by_step = defaultdict(lambda: [0, 0])      # step -> [reached, wrong there]
        joint = defaultdict(lambda: [0, 0])        # (steps 0-4, 5-9, ...; distance bin) -> [written, wrong]
        right_answers = 0
        for task, w in zip(tasks, written):
            line = task.prompt.decode("latin-1")
            assignments = line[:line.rindex(";?")].split(";")
            binding = dict(a.split("=", 1) for a in assignments)
            position = {}
            at = 0
            for a in assignments:
                position[a.split("=", 1)[0]] = at   # where the assignment starts in the line
                at += len(a) + 1
            pointing = {v: k for k, v in binding.items() if not v.isdigit()}
            query = line[line.rindex("?") + 1:-1]
            right = P.chain_written(task).decode("latin-1").rstrip("\n").split("=")
            text = w.decode("latin-1")
            body = text.split("\n", 1)[0].replace("?", "")
            got = body.split("=") if body else [""]
            right_answers += P.grade(task, w) == 1.0
            offset, previous, first = len(line), query, None
            for i, r in enumerate(right):
                g = got[i] if i < len(got) else None
                distance = offset - position[previous]
                wrong = g != r
                b = next(k for k, (lo, hi) in enumerate(BINS) if lo <= distance < hi) if distance < 400 else len(BINS) - 1
                by_distance[b][0] += 1
                by_distance[b][1] += wrong
                by_step[i][0] += 1
                by_step[i][1] += wrong
                joint[(i // 5, b)][0] += 1
                joint[(i // 5, b)][1] += wrong
                if wrong:
                    first = i
                    if g is None:
                        kind = "stopped (newline or room) before this step"
                    elif r.isdigit() and g.isdigit():
                        kind = "a wrong digit at the end"
                    elif r.isdigit():
                        kind = "a name where the digit belongs (went on)"
                    elif g.isdigit():
                        kind = "a digit too early (stopped short)"
                    elif g == binding.get(r):
                        kind = "a skipped step"
                    elif g == previous:
                        kind = "the previous name repeated"
                    elif g == pointing.get(previous):
                        kind = "a step backwards"
                    elif g in binding:
                        kind = "another name of the line"
                    else:
                        kind = "a name not in the line"
                    kinds[kind] += 1
                    break
                offset += len(r) + 1
                previous = r
            if first is None:
                kinds["the whole chain right" if len(got) == len(right) else "right, then more"] += 1
            else:
                first_steps[first] += 1
        found[name][str(hops)] = {
            "right answers (last byte)": right_answers / 200,
            "answers by the first error's kind": dict(kinds.most_common()),
            "answers by the step of the first error": {str(k): v for k, v in sorted(first_steps.items())},
            "steps up to the first error, by the distance back to the assignment they need (bytes): written, wrong":
                {f"{BINS[k][0]}-{BINS[k][1]}": v for k, v in sorted(by_distance.items())},
            "steps up to the first error, by step: reached, wrong there": {str(k): v for k, v in sorted(by_step.items())},
            "steps up to the first error, by step (5 a group) and distance back (bytes): written, wrong":
                {f"steps {5 * k}-{5 * k + 4}, {BINS[b][0]}-{BINS[b][1]}": v for (k, b), v in sorted(joint.items())}}
        print(name, hops, right_answers / 200, dict(kinds.most_common(4)), flush=True)
json.dump(found, out.open("w"), indent=1)
