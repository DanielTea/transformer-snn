"""Package the published spiking byte model and matched study checkpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "event_lm"
RELEASE = "spiking-byte-lm-v1"
ARCHIVE = "spiking-byte-lm-checkpoints.tar.gz"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(directory):
    entries = {}
    candidates = []
    records = sorted(RESULTS.glob("*_s*_threshold_scale_*.json"))
    for record_path in records:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        args = record.get("args", {})
        if not args.get("checkpoint"):
            continue
        name = Path(args["checkpoint"]).name
        entry = entries.setdefault(name, {})
        entry["training_record"] = record_path.relative_to(ROOT).as_posix()
        if args["tag"] == "threshold_scale_large_context":
            candidates.append((record["val_bpb"], name, record, record_path))
        parent = record.get("initialization")
        if parent:
            parent_name = Path(parent["checkpoint"]).name
            entry["parent"] = parent_name
            entries.setdefault(parent_name, {})["expected_sha256"] = parent["sha256"]

    if len(entries) != 39 or len(candidates) != 3:
        raise ValueError("expected 36 matched-study runs, three router parents, and three release candidates")
    for name, entry in entries.items():
        path = directory / name
        entry.update(bytes=path.stat().st_size, sha256=sha256(path))
        expected = entry.pop("expected_sha256", entry["sha256"])
        if expected != entry["sha256"]:
            raise ValueError(f"parent checkpoint hash does not match training metadata: {path}")
    return entries, min(candidates, key=lambda candidate: candidate[:2])


def add_file(archive, name, length, source):
    member = tarfile.TarInfo(name)
    member.size = length
    member.mode = 0o644
    archive.addfile(member, source)


def write_archive(checkpoints, entries, destination):
    lines = [f'{entry["sha256"]}  checkpoints/{name}' for name, entry in sorted(entries.items())]
    checksum_list = ("\n".join(lines) + "\n").encode("utf-8")
    with destination.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            add_file(archive, "checkpoints/SHA256SUMS", len(checksum_list), io.BytesIO(checksum_list))
            for name, entry in sorted(entries.items()):
                with (checkpoints / name).open("rb") as source:
                    add_file(archive, f"checkpoints/{name}", entry["bytes"], source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "spiking-byte-lm")
    args = parser.parse_args()
    try:
        entries, (val_bpb, name, training, record_path) = collect(args.checkpoint_dir)
        audit = json.loads((RESULTS / "threshold_scale_large_activity.json").read_text(encoding="utf-8"))
        measured = next(row for row in audit["models"] if Path(row["checkpoint"]).name == name)
        args.release_dir.mkdir(parents=True, exist_ok=True)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        bundle = args.release_dir / ARCHIVE
        write_archive(args.checkpoint_dir, entries, bundle)
        bundle_sha256 = sha256(bundle)
        (args.release_dir / "spiking-byte-lm-checkpoints.sha256").write_text(
            f"{bundle_sha256}  {bundle.name}\n", encoding="utf-8"
        )
        shutil.copyfile(args.checkpoint_dir / name, args.model_dir / "weights.pt")
        manifest = {
            "release": RELEASE,
            "model": {
                "file": "weights.pt", "source_checkpoint": name, "sha256": entries[name]["sha256"],
                "architecture": training["cfg"], "parameters": training["params"],
                "streaming_tensor_bytes": measured["streaming_tensor_bytes"],
                "training_targets": training["train_target_bytes"], "seed": training["args"]["seed"],
                "selection_validation_bpb": val_bpb,
                "training_record": record_path.relative_to(ROOT).as_posix(),
            },
            "archive": {"file": bundle.name, "sha256": bundle_sha256, "bytes": bundle.stat().st_size},
            "checkpoints": dict(sorted(entries.items())),
        }
        (args.model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Packaged {len(entries)} checkpoints in {bundle}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
