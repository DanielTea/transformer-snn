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
