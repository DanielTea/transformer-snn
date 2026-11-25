"""The human-written language corpus (experiments/language_corpus.py): what it keeps of a textbook module and of a
worked solution, and its splits."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from language_corpus import gsm8k_text, openstax_text, write_corpus  # noqa: E402
from train_event_lm import load_corpus  # noqa: E402

MODULE = """<document xmlns="http://cnx.rice.edu/cnxml"><title>Acceleration</title>
<metadata><md:title>hidden metadata</md:title></metadata><content>
<note class="os-teacher"><para>Only for teachers.</para></note>
<para>Acceleration is a change in velocity &amp; it has a direction, so a car that turns accelerates even at a
steady speed; a long enough paragraph to keep the module, with words that a reader of physics would know well.</para>
<equation><m:math><m:mi>a</m:mi><m:mo>=</m:mo><m:mn>2</m:mn></m:math></equation>
</content></document>"""


def test_a_module_keeps_its_prose_and_drops_the_teachers_notes(tmp_path):
    folder = tmp_path / "modules" / "m1"
    folder.mkdir(parents=True)
    (folder / "index.cnxml").write_text(MODULE)
    (text,) = openstax_text(tmp_path)
    assert text.startswith("Acceleration\n") and "velocity & it has" in text
    assert "teachers" not in text and "metadata" not in text and "a = 2" in text.replace("  ", " ")


def test_a_worked_solution_loses_its_calculator_marks(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"question": "How many?", "answer": "He has 48/2 = <<48/2=24>>24.\n#### 24"}) + "\n")
    (text,) = gsm8k_text(path)
    assert text == "Question: How many?\nHe has 48/2 = 24.\nAnswer: 24\n\n"


def test_the_corpus_splits_by_documents(tmp_path):
    documents = [f"Question: {i}?\nAnswer: {i}\n\n" for i in range(40)]
    sidecar = write_corpus(list(documents), tmp_path / "human.bin", {"test": {}})
    train, validation, test = load_corpus(tmp_path / "human.bin")
    assert sidecar["documents"] == 40 and len(train) + len(validation) + len(test) == sum(len(d) for d in documents)
    assert validation.tobytes().decode().startswith("Question:")
