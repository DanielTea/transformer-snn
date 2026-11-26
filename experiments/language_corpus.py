"""Human-written text for the one-model plan, as a byte corpus: a physics textbook and grade-school worked solutions.

  openstax  OpenStax "Physics" (CC BY 4.0, openstax/osbooks-physics): every module's prose, its titles, paragraphs,
            list items, worked examples and exercises as plain text; the teacher-support notes left out. Equations in
            MathML keep their numbers, names and operators as text. Attribution: "Physics" by OpenStax, CC BY 4.0.
  gsm8k     GSM8K (MIT): every training problem with its human-written step-by-step solution, the calculator marks
            (<<48/2=24>>) taken out, and its final answer.
Documents are shuffled and split into train, validation and test by documents; the corpus is raw UTF-8 bytes with a
sidecar of its split sizes, as train_event_lm.py --corpus reads it. The markup is stripped with regular expressions,
never parsed, since downloaded files are data.

usage: python -I experiments/language_corpus.py --openstax data/onemodel/openstax-physics/src \\
    --gsm8k data/onemodel/gsm8k/train.jsonl --out data/onemodel/language/human-v1.bin
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import re
from pathlib import Path

DROPPED = re.compile(r"<note[^>]*class=\"os-teacher\"[^>]*>.*?</note>|<metadata.*?</metadata>|<media.*?</media>|"
                     r"<figure.*?</figure>", re.S)
BREAKS = re.compile(r"</?(para|title|item|list|section|example|exercise|problem|solution|note|equation|table|row)\b[^>]*>")


def openstax_text(src: Path) -> list[str]:
    """The prose of every module of the book, a document each."""
    documents = []
    for path in sorted(src.glob("modules/*/index.cnxml")):
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = DROPPED.sub(" ", raw)
        raw = re.sub(r"<m:mo>([^<]*)</m:mo>", r" \1 ", raw)
        raw = BREAKS.sub("\n", raw)
        raw = re.sub(r"<[^>]+>", "", raw)
        text = html.unescape(raw)
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        if len(text) > 200:
            documents.append(text + "\n\n")
    return documents


def gsm8k_text(path: Path) -> list[str]:
    documents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        steps, answer = row["answer"].rsplit("####", 1)
        steps = re.sub(r"<<[^>]*>>", "", steps).strip()
        documents.append(f"Question: {row['question'].strip()}\n{steps}\nAnswer: {answer.strip()}\n\n")
    return documents


def write_corpus(documents: list[str], out: Path, sources: dict, seed: int = 1, shares=(0.9, 0.05, 0.05)) -> dict:
    random.Random(seed).shuffle(documents)
    cut1, cut2 = int(shares[0] * len(documents)), int((shares[0] + shares[1]) * len(documents))
    parts = ["".join(documents[:cut1]).encode(), "".join(documents[cut1:cut2]).encode(), "".join(documents[cut2:]).encode()]
    raw = b"".join(parts)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(raw)
    sidecar = {"format": "raw-bytes-1", "task": "human-written language", "documents": len(documents),
               "sources": sources, "seed": seed, "sha256": hashlib.sha256(raw).hexdigest(),
               "train": len(parts[0]), "validation": len(parts[1]), "test": len(parts[2])}
    Path(f"{out}.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--openstax", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    book, solutions = openstax_text(args.openstax), gsm8k_text(args.gsm8k)
    sources = {"openstax_physics": {"documents": len(book), "bytes": sum(len(d.encode()) for d in book),
                                    "licence": "CC BY 4.0", "attribution": "Physics by OpenStax"},
               "gsm8k_train": {"documents": len(solutions), "bytes": sum(len(d.encode()) for d in solutions),
                               "licence": "MIT"}}
    print(json.dumps(write_corpus(book + solutions, args.out, sources)))


if __name__ == "__main__":
    main()
