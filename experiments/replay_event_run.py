"""Replay a saved event language model training run with local paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path


def checked_path(recorded, directory, metadata, role, *, dry_run):
    path = Path(recorded) if directory is None else directory / Path(recorded).name
    expected = (metadata or {}).get("sha256")
    if expected and not dry_run:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected:
            raise ValueError(f"{role} checkpoint checksum differs from the training record: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="training JSON from results/event_lm")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--training-mixture", type=Path, help="override the recorded training mixture catalogue path")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/replayed"))
    parser.add_argument("--parent-checkpoints", type=Path, help="override the directory of a recorded parent checkpoint")
    parser.add_argument("--teacher-checkpoints", type=Path, help="override the directory of a recorded teacher checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        record = json.loads(args.record.read_text(encoding="utf-8"))
        options = record["args"].copy()
        if "train_target_bytes" not in record or "variant" not in options:
            raise ValueError("expected an event language model training record")
        parent = options.pop("initialize_from", None)
        checkpoint = options.get("checkpoint") or f"{args.record.stem}.pt"
        options.update(data=args.enwik8, checkpoint=args.output_dir / Path(checkpoint).name,
                       results_dir=args.output_dir / "results", force=False)
        if args.device:
            options["device"] = args.device
        if args.training_mixture is not None and not options.get("training_mixture"):
            raise ValueError("the saved run trained on enwik8 only")
        if options.get("training_mixture"):
            catalogue = args.training_mixture or Path(options["training_mixture"])
            expected = record["training_mixture"]["catalogue_sha256"]
            if not args.dry_run and hashlib.sha256(catalogue.read_bytes()).hexdigest() != expected:
                raise ValueError(f"training mixture catalogue checksum differs: {catalogue}")
            options["training_mixture"] = catalogue
        if parent:
            options["initialize_from"] = checked_path(
                parent, args.parent_checkpoints, record.get("initialization"), "parent", dry_run=args.dry_run,
            )
        teacher = options.get("teacher_checkpoint")
        if teacher:
            options["teacher_checkpoint"] = checked_path(
                teacher, args.teacher_checkpoints, record.get("teacher"), "teacher", dry_run=args.dry_run,
            )
        command = [sys.executable, str(Path(__file__).with_name("train_event_lm.py"))]
        for name, value in options.items():
            flag = "--" + name.replace("_", "-")
            if value is True:
                command.append(flag)
            elif value is not None and value is not False:
                command.append(f"{flag}={value}")
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
