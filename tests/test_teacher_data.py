"""Teacher-written training text (experiments/teacher_data.py): the checks that decide what is kept, the client
against a stand-in endpoint, and the corpus."""
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import teacher_data as td  # noqa: E402
from train_event_lm import load_corpus  # noqa: E402


def test_the_final_number_is_the_last_answer():
    assert td.final_number("so 3 + 4 = 7.\nAnswer: 1,234") == 1234
    assert td.final_number("Answer: 5 ... wait, Answer: **12.5**") == 12.5
    assert td.final_number("the result is 42") == 42 and td.final_number("no digits") is None


def test_physics_answers_are_the_programs():
    rng = random.Random(0)
    for _ in range(50):
        item = td.physics_item(rng)
        assert td.check_physics(item, f"Answer: {item['answer']:.6g}")[0]
        assert not td.check_physics(item, f"Answer: {item['answer'] * 1.05 + 1:.6g}")[0]
    flow = next(i for i in (td.physics_item(random.Random(s)) for s in range(100)) if i["kind"] == "affinity_flow")
    import re
    q1 = float(re.search(r"delivers (\d+) m3/h", flow["question"]).group(1))
    n1, n2 = (float(x) for x in re.findall(r"(\d+) rpm", flow["question"]))
    assert abs(flow["answer"] - q1 * n2 / n1) < 1e-9


def test_gsm8k_is_kept_only_with_the_right_answer():
    item = {"question": "q", "answer": 72.0}
    assert td.check_gsm8k(item, "Step one... Answer: 72")[0]
    assert not td.check_gsm8k(item, "Answer: 70")[0] and not td.check_gsm8k(item, "I am not sure.")[0]


def test_an_explanation_is_kept_when_its_cause_and_numbers_hold():
    facts = td.fault_facts(np.random.default_rng(3))
    flow = facts["channels"]["flow_m3h"]
    word = td.CAUSES[facts["kind"]][0]
    good = f"The flow went from {flow['before']} m3/h to {flow['after']} m3/h; the likely cause is {word}."
    assert td.check_explain(facts, good)[0]
    assert not td.check_explain(facts, f"The flow went to {flow['after'] * 2 + 50:.1f} m3/h because of {word}.")[0]
    assert not td.check_explain(facts, "Something changed; please look at the pump.")[0]
    assert "flow" in td.facts_text(facts) and facts["start"].count(":") == 1


class _Stub(BaseHTTPRequestHandler):
    replies = []

    def log_message(self, *args):
        pass

    def _send(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"status": "ok"} if self.path == "/health" else {"data": [{"id": "stub-model"}]})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        json.loads(self.rfile.read(length))
        self._send({"choices": [{"message": {"content": _Stub.replies.pop(0)}}]})


def test_a_job_asks_checks_and_keeps_against_an_endpoint(tmp_path, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rng = random.Random(1 + 0)
        items = [td.physics_item(rng) for _ in range(3)]
        _Stub.replies[:] = [f"<think>work it out</think>Step... Answer: {items[0]['answer']:.6g}", "Answer: -1",
                            f"Answer: {items[2]['answer']:.6g}"]
        teacher = td.Teacher(f"http://127.0.0.1:{server.server_port}")
        result = td.run_job("physics", teacher, tmp_path, 3, seed=1, log=lambda m: None)
        assert result == {"job": "physics", "asked": 3, "kept": 2}
        rows = [json.loads(line) for line in (tmp_path / "physics.jsonl").read_text().splitlines()]
        assert [r["kept"] for r in rows] == [True, False, True] and rows[0]["thinking"] == "work it out"
        assert rows[0]["model"] == "stub-model" and "<think>" not in rows[0]["reply"]
        corpus = td.write_corpus(tmp_path, tmp_path / "teacher.bin", shares=(0.5, 0.5, 0.0))
        train, validation, test = load_corpus(tmp_path / "teacher.bin")
        assert corpus["items"] == 2 and len(train) + len(validation) + len(test) == len((tmp_path / "teacher.bin").read_bytes())
    finally:
        server.shutdown()


def test_a_job_stops_at_its_time():
    import datetime as dt
    teacher = td.Teacher("http://127.0.0.1:9", stop=dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=1))
    assert not teacher.time_left()


def test_explains_draws_are_unchanged_and_diagnoses_windows_hold_the_speed():
    # The same draws as before for explain (steady off), so that its items stay reproducible.
    assert td.fault_facts(np.random.default_rng(3)) == td.fault_facts(np.random.default_rng(3), steady=False)
    rng = np.random.default_rng(11)
    for _ in range(40):
        facts = td.fault_facts(rng, steady=True)
        speed = facts["channels"]["speed_rpm"]
        assert speed["before"] > 100 and abs(speed["change"]) < 0.01 * facts["rated_rpm"]
        assert speed["spread_before"] < 0.01 * facts["rated_rpm"] and speed["spread_after"] < 0.01 * facts["rated_rpm"]
    assert td.fault_facts(np.random.default_rng(5), steady=True, kind="seal_leak")["kind"] == "seal_leak"


@pytest.mark.parametrize("line, said", [
    ("Cause: low suction", ("low suction", None)),
    ("**Cause:** Bearing wear", ("bearing wear", None)),
    ("Cause: valve stuck shut.", ("valve stuck shut", None)),
    ("Cause: sensor drift on the flow sensor", ("sensor drift", "flow_m3h")),
    ("cause: frozen sensor on winding", ("frozen sensor", "winding_c")),
    ("Cause: low suction (a falling tank or a clogging strainer)", ("low suction", None)),
    ("Cause: frozen sensor", None),
    ("Cause: sensor drift on speed", None),
    ("Cause: low suction or seal leak", None),
    ("Cause: cavitation", None),
    ("The cause is misalignment.", None),
])
def test_the_cause_is_read_from_the_last_line_only(line, said):
    assert td.diagnosis(f"It is not cavitation and not a seal leak.\n\nCheck the suction first.\n{line}\n") == said
    assert td.diagnosis(f"{line}\nMore words after the line.") is None


def test_a_diagnosis_is_kept_only_for_the_faults_cause_and_numbers_that_hold():
    for kind in td.FAULTS:
        facts = td.fault_facts(np.random.default_rng(7), steady=True, kind=kind)
        name = next(n for n, kinds in td.DIAGNOSES.items() if kind in kinds)
        sensor = f" on {facts['channel'].split('_')[0]}" if facts["channel"] else ""
        flow = facts["channels"]["flow_m3h"]
        body = f"Flow went from {flow['before']} m3/h to {flow['after']} m3/h at rated {facts['rated_rpm']:.0f} rpm.\n"
        assert td.check_diagnose(facts, body + f"Cause: {name}{sensor}")[0], kind
        wrong = next(n for n in td.DIAGNOSES if kind not in td.DIAGNOSES[n] and not n.startswith(("sensor", "frozen")))
        assert not td.check_diagnose(facts, body + f"Cause: {wrong}")[0]
        assert not td.check_diagnose(facts, f"Flow is {flow['after'] * 2 + 50:.1f} m3/h.\nCause: {name}{sensor}")[0]
        if facts["channel"]:
            other = next(s for s, c in td.SENSORS.items() if c != facts["channel"])
            assert not td.check_diagnose(facts, body + f"Cause: {name} on {other}")[0]
            assert not td.check_diagnose(facts, body + f"Cause: {name}")[0]
    # Both suction faults are one cause, and the differential pressure counts as measured.
    for kind in ("cavitation", "clogging"):
        facts = td.fault_facts(np.random.default_rng(2), steady=True, kind=kind)
        rows = facts["channels"]
        lift = rows["discharge_bar"]["after"] - rows["suction_bar"]["after"]
        assert td.check_diagnose(facts, f"The pump now lifts {lift:.2f} bar.\nCause: low suction")[0]


def test_a_diagnose_job_runs_against_an_endpoint(tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        rng = np.random.default_rng(1 + 0)
        items = [td.fault_facts(rng, steady=True) for _ in range(2)]
        name = lambda f: next(n for n, kinds in td.DIAGNOSES.items() if f["kind"] in kinds)  # noqa: E731
        sensor = lambda f: f" on {f['channel'].split('_')[0]}" if f["channel"] else ""  # noqa: E731
        _Stub.replies[:] = [f"Reasoning in plain words.\nCause: {name(items[0])}{sensor(items[0])}",
                            "Something changed.\nCause: none of these"]
        teacher = td.Teacher(f"http://127.0.0.1:{server.server_port}")
        assert td.run_job("diagnose", teacher, tmp_path, 2, seed=1, log=lambda m: None) == {
            "job": "diagnose", "asked": 2, "kept": 1}
        rows = [json.loads(line) for line in (tmp_path / "diagnose.jsonl").read_text().splitlines()]
        assert rows[0]["item"] == items[0] and rows[0]["system"] == td.DIAGNOSER and "Cause: <cause>" in td.DIAGNOSER
        corpus = td.write_corpus(tmp_path, tmp_path / "teacher.bin", shares=(1.0, 0.0, 0.0))
        assert corpus["items"] == 1 and b"Telemetry:" in (tmp_path / "teacher.bin").read_bytes()
    finally:
        server.shutdown()
