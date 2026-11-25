"""Teacher-written training text for the one-model plan, checked before it is kept.

A chat model behind an OpenAI-compatible endpoint (the DeepSeek that the shared server serves by day; Daniel,
2026-11-25: "deepseek is fine") writes text that the spiking model later trains on; the model that is trained and
released stays purely spiking. Every reply is checked by a program before it is kept, and the teacher's model id goes
with every item. Three jobs:

  gsm8k    step-by-step solutions to the GSM8K training problems (human-written problems, MIT); kept when the reply's
           final answer equals the problem's
  physics  pump problems drawn by a program (affinity laws, hydraulic and shaft power, motor current, NPSH margin)
           with answers the program computes; kept when the reply's final number is within 1% of it
  explain  explanations for an operator of simulated pump faults (experiments/pump_sim.py), from the measured facts
           of each fault; kept when the fact checker finds the fault's cause named and every number given with a
           channel's unit within 5% of a measured value or change

The endpoint is asked one request at a time, only while its /health answers ok, and the job ends at --stop-utc on its
own, so that it holds no request when the scheduler stops the server's model to open a GPU window.
Output: OUT/JOB.jsonl, a line per item (prompt, reply, thinking if the model returns it, verdict and reason, model id,
time). ``corpus`` turns the kept items of all jobs into a byte corpus with a sidecar, as train_event_lm.py --corpus
reads it.

usage:
  python experiments/teacher_data.py run --job gsm8k --base http://127.0.0.1:18000 --out data/onemodel/teacher \\
      --count 500 --stop-utc 16:20
  python experiments/teacher_data.py corpus --inputs data/onemodel/teacher --out data/onemodel/teacher/teacher-v1.bin
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pump_sim import CHANNELS, FAULTS, Fault, Pump, Run, simulate  # noqa: E402

GSM8K = Path(__file__).resolve().parents[1] / "data" / "onemodel" / "gsm8k" / "train.jsonl"
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
UNITS = {"speed_rpm": ("rpm",), "flow_m3h": ("m3/h", "m³/h", "m^3/h"), "suction_bar": ("bar",),
         "discharge_bar": ("bar",), "current_a": ("A", "amps", "amperes"), "vibration_mms": ("mm/s",),
         "bearing_c": ("°C", "C"), "winding_c": ("°C", "C")}
CAUSES = {"cavitation": ("cavitat",), "clogging": ("clog", "strainer", "blocked suction", "suction blockage"),
          "bearing_wear": ("bearing",), "imbalance": ("imbalance", "unbalance"), "misalignment": ("misalign",),
          "valve_stuck": ("valve",), "seal_leak": ("seal", "leak"), "sensor_drift": ("drift", "calibrat"),
          "sensor_freeze": ("frozen", "stuck", "flatline", "flat line", "not updating")}


class Teacher:
    """A chat endpoint asked one request at a time, with retries and the time limit of the job."""

    def __init__(self, base: str, key: str | None = None, stop: dt.datetime | None = None, timeout: float = 900.0,
                 max_tokens: int = 2000):
        self.base, self.key, self.stop, self.timeout = base.rstrip("/"), key, stop, timeout
        self.max_tokens = max_tokens
        self.model = None

    def _call(self, path: str, body: dict | None = None):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = urllib.request.Request(self.base + path, headers=headers,
                                         data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def healthy(self) -> bool:
        try:
            return self._call("/health").get("status") == "ok"
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def time_left(self) -> bool:
        return self.stop is None or dt.datetime.now(dt.timezone.utc) < self.stop

    def model_id(self) -> str:
        if self.model is None:
            try:
                self.model = self._call("/v1/models")["data"][0]["id"]
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
                self.model = "unknown"
        return self.model

    def ask(self, system: str, user: str, *, temperature: float = 0.7, tries: int = 3) -> tuple[str, str]:
        """The reply and the model's separate reasoning (empty if it gives none)."""
        body = {"model": self.model_id(), "temperature": temperature, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        for attempt in range(tries):
            try:
                message = self._call("/v1/chat/completions", body)["choices"][0]["message"]
                reply, thinking = message.get("content") or "", message.get("reasoning_content") or ""
                found = re.search(r"<think>(.*?)</think>", reply, re.S)
                if found:
                    thinking, reply = found.group(1).strip(), (reply[:found.start()] + reply[found.end():]).strip()
                return reply.strip(), thinking.strip()
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError):
                if attempt == tries - 1:
                    raise
                time.sleep(30 * (attempt + 1))
        raise RuntimeError("unreachable")


def final_number(text: str) -> float | None:
    """The number after the last 'Answer:' of a reply, or its last number."""
    found = re.findall(r"answer\s*[:=]\s*\**\s*\$?\s*(" + NUMBER + ")", text, re.I)
    if not found:
        found = re.findall(NUMBER, text)
    if not found:
        return None
    try:
        return float(found[-1].replace(",", ""))
    except ValueError:
        return None


# --- gsm8k -----------------------------------------------------------------------------------------------------

SOLVER = ("You solve problems by reasoning step by step in plain words, then give the result on a last line of the form "
          "'Answer: <number>'.")


def gsm8k_items(path: Path = GSM8K):
    for line in path.read_text().splitlines():
        row = json.loads(line)
        yield {"question": row["question"], "answer": float(row["answer"].rsplit("####", 1)[1].strip().replace(",", ""))}


def check_gsm8k(item: dict, reply: str) -> tuple[bool, str]:
    got = final_number(reply)
    if got is None:
        return False, "no final number"
    return math.isclose(got, item["answer"], rel_tol=1e-9, abs_tol=1e-9), f"answer {got} for {item['answer']}"


# --- physics -----------------------------------------------------------------------------------------------------

def physics_item(rng: random.Random) -> dict:
    """A pump problem with its exact answer, of one of five kinds."""
    kind = rng.choice(("affinity_flow", "affinity_power", "hydraulic_power", "motor_current", "npsh_margin"))
    if kind == "affinity_flow":
        q1, n1, n2 = rng.randint(40, 400), rng.choice((1475, 2950)), rng.randint(900, 3000)
        text = f"A centrifugal pump delivers {q1} m3/h at {n1} rpm. What flow does it deliver at {n2} rpm, in m3/h?"
        answer = q1 * n2 / n1
    elif kind == "affinity_power":
        p1, n1, n2 = rng.randint(5, 150), rng.choice((1475, 2950)), rng.randint(900, 3000)
        text = (f"A pump draws {p1} kW of shaft power at {n1} rpm. By the affinity laws, what shaft power does it draw "
                f"at {n2} rpm, in kW?")
        answer = p1 * (n2 / n1) ** 3
    elif kind == "hydraulic_power":
        q, h, eta = rng.randint(40, 400), rng.randint(10, 90), rng.randint(55, 85)
        text = (f"Water (density 1000 kg/m3, g = 9.81 m/s2) is pumped at {q} m3/h against a head of {h} m by a pump of "
                f"{eta}% efficiency. What is the shaft power in kW?")
        answer = 1000 * 9.81 * (q / 3600) * h / (eta / 100) / 1000
    elif kind == "motor_current":
        p, v, pf, eff = rng.randint(5, 150), rng.choice((400, 690)), rng.randint(80, 92), rng.randint(88, 96)
        text = (f"A three-phase motor of {eff}% efficiency on {v} V with a power factor of 0.{pf} delivers {p} kW to its "
                f"shaft. What current does it draw, in amperes?")
        answer = p * 1000 / (math.sqrt(3) * v * (pf / 100) * (eff / 100))
    else:
        level, loss, pv, npshr = rng.randint(1, 12), round(rng.uniform(0.3, 3.0), 1), round(rng.uniform(0.02, 0.2), 2), \
            round(rng.uniform(2.0, 6.0), 1)
        text = (f"A pump takes water from an open tank whose surface is {level} m above the pump inlet. Atmospheric "
                f"pressure is 1.013 bar, the vapour pressure is {pv} bar, and the suction line loses {loss} m of head. "
                f"With density 1000 kg/m3 and g = 9.81 m/s2, what is the margin of the available NPSH over the "
                f"{npshr} m the pump requires, in m?")
        answer = (1.013 - pv) * 1e5 / (1000 * 9.81) + level - loss - npshr
    return {"kind": kind, "question": text, "answer": answer}


def check_physics(item: dict, reply: str) -> tuple[bool, str]:
    got = final_number(reply)
    if got is None:
        return False, "no final number"
    ok = math.isclose(got, item["answer"], rel_tol=0.01, abs_tol=0.02)
    return ok, f"answer {got} for {item['answer']:.4g}"


# --- explain -----------------------------------------------------------------------------------------------------

EXPLAINER = ("You are a reliability engineer. From the telemetry facts you are given, write to the operator in plain "
             "words: what changed, the most likely cause and why, how urgent it is, and what to check first. Use only "
             "the numbers in the facts.")


def fault_facts(rng: np.random.Generator, hours: float = 4.0) -> dict:
    """A simulated pump with one fault, and the measured facts of it: the channels' means over the ten minutes before
    the fault and the ten minutes after it is at full strength."""
    seconds = int(hours * 3600)
    kind = str(rng.choice(FAULTS))
    channel = str(rng.choice(CHANNELS[1:])) if kind.startswith("sensor") else None
    ramp = 1 if kind == "sensor_freeze" else int(rng.choice([60, 600, 1800]))
    start = int(rng.integers(seconds // 3, seconds // 2))
    run = Run(Pump.draw(rng), seconds, [Fault(kind, start, ramp, float(rng.uniform(0.6, 1.0)), channel)],
              seed=int(rng.integers(2 ** 31)))
    values = simulate(run)
    before = values[max(0, start - 600):start].mean(0)
    after = values[min(seconds - 600, start + ramp):min(seconds, start + ramp + 600)].mean(0)
    spread = values[max(0, start - 600):start].std(0)
    after_spread = values[min(seconds - 600, start + ramp):min(seconds, start + ramp + 600)].std(0)
    rows = {}
    for i, name in enumerate(CHANNELS):
        rows[name] = {"before": round(float(before[i]), 2), "after": round(float(after[i]), 2),
                      "change": round(float(after[i] - before[i]), 2),
                      "spread_before": round(float(spread[i]), 3), "spread_after": round(float(after_spread[i]), 3)}
    clock = lambda s: f"{s // 3600:02d}:{s % 3600 // 60:02d}"  # noqa: E731
    return {"kind": kind, "channel": channel, "start": clock(start), "ramp_s": ramp, "channels": rows,
            "rated_rpm": run.pump.n0}


def facts_text(facts: dict) -> str:
    lines = [f"Pump rated {facts['rated_rpm']:.0f} rpm. A change began at {facts['start']} (hh:mm) and took about "
             f"{facts['ramp_s']} s to settle. Means over the ten minutes before and after:"]
    for name, row in facts["channels"].items():
        unit = UNITS[name][0]
        lines.append(f"- {name.split('_')[0]}: {row['before']} -> {row['after']} {unit} (change {row['change']:+}; "
                     f"spread {row['spread_before']} -> {row['spread_after']})")
    return "\n".join(lines)


def check_explain(facts: dict, reply: str) -> tuple[bool, str]:
    """The fault's cause is named, and every number given with a channel's unit is within 5% of a measured value
    or change of a channel with that unit (or within 0.05 of it)."""
    text = reply.lower()
    if not any(word in text for word in CAUSES[facts["kind"]]):
        return False, f"cause {facts['kind']} not named"
    measured = {}
    for name, row in facts["channels"].items():
        for unit in UNITS[name]:
            measured.setdefault(unit.lower(), []).extend(
                [row["before"], row["after"], abs(row["change"]), row["spread_before"], row["spread_after"]])
    for number, unit in re.findall(r"(" + NUMBER + r")\s*(m3/h|m³/h|m\^3/h|rpm|bar|mm/s|°c|amps|amperes|a)\b",
                                   text):
        value = abs(float(number.replace(",", "")))
        allowed = measured.get(unit, [])
        if allowed and not any(math.isclose(value, abs(m), rel_tol=0.05, abs_tol=0.05) for m in allowed):
            return False, f"{number} {unit} matches no measured value"
    return True, "cause named; numbers match"


# --- the job -----------------------------------------------------------------------------------------------------

def run_job(job: str, teacher: Teacher, out: Path, count: int, seed: int, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{job}.jsonl"
    done = sum(1 for _ in path.open()) if path.exists() else 0
    kept = asked = 0
    rng, nrng = random.Random(seed + done), np.random.default_rng(seed + done)
    if job == "gsm8k":
        items = list(gsm8k_items())[done:done + count]
    with path.open("a") as handle:
        for index in range(count):
            if not teacher.time_left():
                log("stop time reached")
                break
            while not teacher.healthy():
                if not teacher.time_left():
                    break
                log("the endpoint is not serving: waiting")
                time.sleep(300)
            if not teacher.time_left():
                break
            if job == "gsm8k":
                if index >= len(items):
                    break
                item = items[index]
                system, user, check = SOLVER, item["question"], check_gsm8k
            elif job == "physics":
                item = physics_item(rng)
                system, user, check = SOLVER, item["question"], check_physics
            else:
                item = fault_facts(nrng)
                system, user, check = EXPLAINER, facts_text(item), check_explain
            started = time.time()
            reply, thinking = teacher.ask(system, user)
            asked += 1
            ok, reason = check(item, reply)
            kept += ok
            handle.write(json.dumps({"job": job, "item": item, "system": system, "prompt": user, "reply": reply,
                                     "thinking": thinking, "kept": ok, "reason": reason, "model": teacher.model_id(),
                                     "seconds": round(time.time() - started, 1),
                                     "utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}) + "\n")
            handle.flush()
            log(f"{job} {done + index + 1}: {'kept' if ok else 'dropped'} ({reason}) in {time.time() - started:.0f} s")
    return {"job": job, "asked": asked, "kept": kept}


def write_corpus(inputs: Path, out: Path, seed: int = 1, shares=(0.9, 0.05, 0.05)) -> dict:
    """The kept items of every job as text, shuffled and split into train, validation and test by items."""
    texts = []
    for path in sorted(inputs.glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if not row["kept"]:
                continue
            if row["job"] == "explain":
                texts.append(f"Telemetry:\n{row['prompt']}\nWhat happened, and what should I do?\n{row['reply']}\n\n")
            else:
                texts.append(f"Question: {row['prompt']}\n{row['reply']}\n\n")
    random.Random(seed).shuffle(texts)
    cut1, cut2 = int(shares[0] * len(texts)), int((shares[0] + shares[1]) * len(texts))
    parts = ["".join(texts[:cut1]).encode(), "".join(texts[cut1:cut2]).encode(), "".join(texts[cut2:]).encode()]
    raw = b"".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    sidecar = {"format": "raw-bytes-1", "task": "teacher text", "items": len(texts), "seed": seed,
               "sha256": hashlib.sha256(raw).hexdigest(), "train": len(parts[0]), "validation": len(parts[1]),
               "test": len(parts[2])}
    Path(f"{out}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="what", required=True)
    run = sub.add_parser("run")
    run.add_argument("--job", choices=("gsm8k", "physics", "explain"), required=True)
    run.add_argument("--base", required=True, help="the endpoint, e.g. http://127.0.0.1:18000")
    run.add_argument("--key-file", type=Path, help="a file holding the endpoint's key, if it wants one")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--count", type=int, default=100)
    run.add_argument("--seed", type=int, default=1)
    run.add_argument("--stop-utc", help="HH:MM today (UTC) after which no request is sent")
    run.add_argument("--max-tokens", type=int, default=2000,
                     help="the reply's limit, a reasoning model's thinking included")
    corpus = sub.add_parser("corpus")
    corpus.add_argument("--inputs", type=Path, required=True)
    corpus.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.what == "corpus":
        print(json.dumps(write_corpus(args.inputs, args.out)))
        return
    stop = None
    if args.stop_utc:
        hour, minute = map(int, args.stop_utc.split(":"))
        stop = dt.datetime.now(dt.timezone.utc).replace(hour=hour, minute=minute, second=0, microsecond=0)
    key = args.key_file.read_text().strip() if args.key_file else os.environ.get("TEACHER_KEY")
    teacher = Teacher(args.base, key, stop, max_tokens=args.max_tokens)
    print(json.dumps(run_job(args.job, teacher, args.out, args.count, args.seed, log=lambda m: print(m, flush=True))))


if __name__ == "__main__":
    main()
