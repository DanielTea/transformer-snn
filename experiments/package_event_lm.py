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
RELEASES = {
    "v1": ("spiking-byte-lm", "spiking-byte-lm-checkpoints"),
    "v2": ("spiking-byte-lm-v2", "spiking-byte-lm-v2-checkpoints"),
    "v3": ("spiking-byte-lm-v3", "spiking-byte-lm-v3-checkpoints"),
    "v4": ("spiking-byte-lm-v4", "spiking-byte-lm-v4-checkpoints"),
    "v5": ("spiking-byte-lm-v5", "spiking-byte-lm-v5-checkpoints"),
    "v6": ("spiking-byte-lm-v6", "spiking-byte-lm-v6-checkpoints"),
    "v7": ("spiking-byte-lm-v7", "spiking-byte-lm-v7-checkpoints"),
    "v8": ("spiking-byte-lm-v8", "spiking-byte-lm-v8-checkpoints"),
    "v9": ("spiking-byte-lm-v9", "spiking-byte-lm-v9-checkpoints"),
    "v10": ("spiking-byte-lm-v10", "spiking-byte-lm-v10-checkpoints"),
}
DEPTH_REFERENCE = "depth-inhibition-study/trace_s0.pt"
CONTINUATION_REFERENCE = "long-context-continuation/trace_ctx2048_s711.pt"
FINE_CHARGE_REFERENCE = "grown-fine-charge-study/fine_s711.pt"
EXPOSURE_REFERENCE = "scale-6510k-next-data-study/trace_s2503.pt"
DEPTH_GROWTH_REFERENCE = "depth-growth-study/depth_s4603.pt"
TRACE_TAP_REFERENCE = "trace-tap-study/control_s6899.pt"
TRACE_READ_REFERENCE = "trace-read-drive-study/write_rails_share_s8237.pt"
TRACE_ARRIVAL_REFERENCE = "trace-arrival-study/unmatched_s9349.pt"
V4_CATALOGS = ("continuation_checkpoints.sha256", "v4_checkpoints.sha256")
V6_MANIFEST = "models/spiking-byte-lm-v6/manifest.json"
V7_MANIFEST = "models/spiking-byte-lm-v7/manifest.json"
V8_MANIFEST = "models/spiking-byte-lm-v8/manifest.json"
ARRIVAL_TRANSFER = "byte_transfer/trace_arrival_s9349.json"
V9_MANIFEST = "models/spiking-byte-lm-v9/manifest.json"
RESIDUAL_SCALE_REFERENCE = "residual-scale-study/wide-snn_s9631.pt"
RESIDUAL_SCALE_TRANSFER = "byte_transfer/residual_scale_131072k.json"


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


def collect_depth_release(directory):
    entries = {}
    records = {}
    for catalog in sorted(RESULTS.glob("*/checkpoints.sha256")):
        for line in catalog.read_text(encoding="utf-8").splitlines():
            expected, _, source = line.partition("  ")
            path = Path(source)
            if path.is_absolute() or ".." in path.parts or path.suffix != ".pt":
                raise ValueError(f"invalid checkpoint path in {catalog}: {source}")
            name = path.relative_to("checkpoints").as_posix()
            if name in entries:
                raise ValueError(f"duplicate checkpoint in release catalogs: {source}")
            entries[name] = {"sha256": expected}
        for record_path in sorted(catalog.parent.glob("*.json")):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                continue
            checkpoint = record.get("args", {}).get("checkpoint")
            if checkpoint:
                records[Path(checkpoint).relative_to("checkpoints").as_posix()] = record, record_path

    prior = json.loads((ROOT / "models/spiking-byte-lm/manifest.json").read_text(encoding="utf-8"))
    for name, entry in list(entries.items()):
        record, record_path = records[name]
        entry["training_record"] = record_path.relative_to(ROOT).as_posix()
        parent = record.get("initialization")
        if parent:
            parent_name = Path(parent["checkpoint"]).relative_to("checkpoints").as_posix()
            if parent_name not in entries:
                entries[parent_name] = prior["checkpoints"][parent_name].copy()
            if entries[parent_name]["sha256"] != parent["sha256"]:
                raise ValueError(f"parent checkpoint hash does not match training metadata: {parent_name}")
            entry["parent"] = parent_name

    for name, entry in entries.items():
        checkpoint = directory / name
        if sha256(checkpoint) != entry["sha256"]:
            raise ValueError(f"checkpoint hash does not match release catalog: {checkpoint}")
        entry["bytes"] = checkpoint.stat().st_size

    record, record_path = records[DEPTH_REFERENCE]
    audit = json.loads((RESULTS / "depth_inhibition/native_deep_trace.json").read_text(encoding="utf-8"))
    measured = next(row for row in audit["models"] if row["checkpoint"] == f"checkpoints/{DEPTH_REFERENCE}")
    return entries, (record["val_bpb"], DEPTH_REFERENCE, record, record_path), measured


def collect_catalog_release(directory, catalogs, reference, audit_path, *, upstream_manifest=None):
    hashes = {}
    for filename in catalogs:
        catalog = RESULTS / filename
        for line in catalog.read_text(encoding="utf-8").splitlines():
            digest, separator, source = line.partition("  ")
            path = Path(source)
            if (not separator or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                    or path.is_absolute() or ".." in path.parts or path.suffix != ".pt" or source in hashes):
                raise ValueError(f"invalid checkpoint entry in {catalog}: {line}")
            path.relative_to("checkpoints")
            hashes[source] = digest

    upstream_hashes = {}
    if upstream_manifest:
        upstream = json.loads((ROOT / upstream_manifest).read_text(encoding="utf-8"))
        upstream_hashes = {f"checkpoints/{name}": entry["sha256"] for name, entry in upstream["checkpoints"].items()}

    records = {}
    for path in sorted(RESULTS.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or "train_target_bytes" not in record:
            continue
        source = (record.get("args") or {}).get("checkpoint")
        if source in hashes:
            if source in records:
                raise ValueError(f"duplicate training records for {source}")
            records[source] = record, path

    entries = {}
    for source, expected in sorted(hashes.items()):
        name = Path(source).relative_to("checkpoints").as_posix()
        checkpoint = directory / name
        if sha256(checkpoint) != expected:
            raise ValueError(f"checkpoint hash does not match release catalog: {checkpoint}")
        record, path = records[source]
        entry = {"bytes": checkpoint.stat().st_size, "sha256": expected,
                 "training_record": path.relative_to(ROOT).as_posix()}
        parent = record.get("initialization")
        if parent:
            parent_path = parent["checkpoint"]
            if parent_path == "models/spiking-byte-lm-v2/weights.pt":
                parent_digest = sha256(ROOT / parent_path)
            else:
                parent_digest = hashes.get(parent_path, upstream_hashes.get(parent_path))
            if parent_digest != parent["sha256"]:
                raise ValueError(f"parent checkpoint is missing or has the wrong hash: {parent_path}")
            entry["parent"] = parent_path
        teacher = record.get("teacher")
        if teacher:
            teacher_path = teacher["checkpoint"]
            if hashes.get(teacher_path) != teacher["sha256"]:
                raise ValueError(f"teacher checkpoint is missing or has the wrong hash: {teacher_path}")
            entry["teacher"] = teacher_path
        entries[name] = entry

    source = f"checkpoints/{reference}"
    record, path = records[source]
    audit = json.loads((RESULTS / audit_path).read_text(encoding="utf-8"))
    measured = next(row for row in audit["models"] if row["checkpoint"] == source)
    return entries, (record["val_bpb"], reference, record, path), measured


def transfer_evidence(path, entries, endpoints):
    record = json.loads(path.read_text(encoding="utf-8"))
    evaluated = {Path(row["checkpoint"]).relative_to("checkpoints").as_posix(): row["checkpoint_sha256"]
                 for row in record["models"]}
    if evaluated != {name: entries[name]["sha256"] for name in endpoints}:
        raise ValueError(f"{path} must score the released study endpoints")
    if any(set(row["benchmarks"]) != {"wikitext103", "c4", "enwik9", "text8", "lambada"} for row in record["models"]):
        raise ValueError(f"{path} is missing a standard transfer benchmark")
    sources = RESULTS / "byte_transfer/sources_v1.json"
    catalogue = RESULTS / "byte_transfer/prepared_v1.json"
    if (sha256(sources) != record["manifest_sha256"] or sha256(catalogue) != record["catalogue_sha256"]):
        raise ValueError(f"{path} uses a different transfer corpus")
    return {"suite": record["suite"], "record": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
            "sources": sources.relative_to(ROOT).as_posix(), "catalogue": catalogue.relative_to(ROOT).as_posix()}


def residual_scale_mixture(training_paths):
    folder = RESULTS / "training_mixture"
    plan = folder / "windows_s9631.json"
    catalogue = folder / "prepared_v1.json"
    sources = folder / "sources_v1.json"
    expected = json.loads(plan.read_text())
    prepared = json.loads(catalogue.read_text())
    if (sha256(catalogue) != expected["catalogue_sha256"]
            or sha256(sources) != prepared["source_manifest_sha256"]):
        raise ValueError("the residual scale study must use its pinned training mixture")
    mixed, control = [], []
    for path in training_paths:
        record = json.loads(path.read_text())
        checkpoint = record["args"]["checkpoint"]
        if Path(checkpoint).name == "enwik8-snn_s9631.pt":
            if record.get("training_mixture") or record["args"].get("training_mixture"):
                raise ValueError(f"the enwik8-only control used the mixture: {path}")
            control.append(checkpoint)
        else:
            if (record.get("training_mixture") != expected
                    or sum(expected["source_targets"].values()) != record["train_target_bytes"]):
                raise ValueError(f"the training windows differ from the matched mixture: {path}")
            mixed.append(checkpoint)
    if len(mixed) != 4 or len(set(mixed)) != 4 or len(control) != 1:
        raise ValueError("the residual scale study needs four matched mixture arms and the enwik8 control")
    return {
        "catalogue": catalogue.relative_to(ROOT).as_posix(), "sources": sources.relative_to(ROOT).as_posix(),
        "window_receipt": plan.relative_to(ROOT).as_posix(), "receipt": expected,
        "mixture_checkpoints": sorted(mixed), "enwik8_only_checkpoint": control[0],
    }


def add_file(archive, name, length, source):
    member = tarfile.TarInfo(name)
    member.size = length
    member.mode = 0o644
    archive.addfile(member, source)


def write_archive(checkpoints, entries, destination, checksum_path):
    lines = [f'{entry["sha256"]}  checkpoints/{name}' for name, entry in sorted(entries.items())]
    checksum_list = ("\n".join(lines) + "\n").encode("utf-8")
    with destination.open("wb") as stream, gzip.GzipFile(filename="", fileobj=stream, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            add_file(archive, checksum_path, len(checksum_list), io.BytesIO(checksum_list))
            for name, entry in sorted(entries.items()):
                with (checkpoints / name).open("rb") as source:
                    add_file(archive, f"checkpoints/{name}", entry["bytes"], source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=RELEASES, default="v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, help="defaults to the selected release's directory in models/")
    args = parser.parse_args()
    try:
        model_dirname, archive_stem = RELEASES[args.release]
        model_dir = args.model_dir or ROOT / "models" / model_dirname
        upstream_manifest = None
        transfer = None
        training_mixture = None
        if args.release == "v1":
            entries, candidate = collect(args.checkpoint_dir)
            audit = json.loads((RESULTS / "threshold_scale_large_activity.json").read_text(encoding="utf-8"))
            measured = next(row for row in audit["models"] if Path(row["checkpoint"]).name == candidate[1])
        elif args.release == "v2":
            entries, candidate, measured = collect_depth_release(args.checkpoint_dir)
        elif args.release == "v3":
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, ("continuation_checkpoints.sha256",), CONTINUATION_REFERENCE,
                "long_context_continuation/native_trace_ctx2048_s711.json",
            )
        elif args.release == "v4":
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, V4_CATALOGS, FINE_CHARGE_REFERENCE, "grown_fine_charge/native_fine_s711.json",
            )
        elif args.release == "v5":
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, (*V4_CATALOGS, "v5_checkpoints.sha256"),
                EXPOSURE_REFERENCE, "scale_6510k_next_data/native_trace_s2503.json",
            )
        elif args.release == "v6":
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, ("v6_checkpoints.sha256",),
                DEPTH_GROWTH_REFERENCE, "depth_growth/native_depth_s4603.json",
            )
        elif args.release == "v7":
            upstream_manifest = V6_MANIFEST
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, ("trace_integration_checkpoints.sha256", "v7_checkpoints.sha256"),
                TRACE_TAP_REFERENCE, "trace_tap_read/native_control_s6899.json", upstream_manifest=upstream_manifest,
            )
        elif args.release == "v8":
            upstream_manifest = V7_MANIFEST
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir,
                ("v8_supporting_checkpoints.sha256", "packed_width_checkpoints.sha256",
                 "trace_read_drive_checkpoints.sha256"),
                TRACE_READ_REFERENCE, "trace_read_drive/native_write_rails_share_s8237.json",
                upstream_manifest=upstream_manifest,
            )
        elif args.release == "v9":
            upstream_manifest = V8_MANIFEST
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, ("trace_arrival_checkpoints.sha256",), TRACE_ARRIVAL_REFERENCE,
                "trace_arrival/native_unmatched_s9349.json", upstream_manifest=upstream_manifest,
            )
            endpoints = [name for name in entries if name.startswith("trace-arrival-study/")]
            transfer = transfer_evidence(RESULTS / ARRIVAL_TRANSFER, entries, endpoints)
        else:
            upstream_manifest = V9_MANIFEST
            entries, candidate, measured = collect_catalog_release(
                args.checkpoint_dir, ("residual_scale_checkpoints.sha256",), RESIDUAL_SCALE_REFERENCE,
                "residual_scale/native_wide-snn_s9631.json", upstream_manifest=upstream_manifest,
            )
            endpoints = [name for name in entries if name.startswith("residual-scale-study/")]
            transfer = transfer_evidence(RESULTS / RESIDUAL_SCALE_TRANSFER, entries, endpoints)
            training_mixture = residual_scale_mixture(ROOT / entries[name]["training_record"] for name in endpoints)
        val_bpb, name, training, record_path = candidate
        args.release_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)
        bundle = args.release_dir / f"{archive_stem}.tar.gz"
        continuation = args.release not in ("v1", "v2")
        checksum_path = f"checkpoints/SHA256SUMS-{args.release}" if continuation else "checkpoints/SHA256SUMS"
        write_archive(args.checkpoint_dir, entries, bundle, checksum_path)
        bundle_sha256 = sha256(bundle)
        (args.release_dir / f"{archive_stem}.sha256").write_text(
            f"{bundle_sha256}  {bundle.name}\n", encoding="utf-8"
        )
        shutil.copyfile(args.checkpoint_dir / name, model_dir / "weights.pt")
        manifest = {
            "release": f"spiking-byte-lm-{args.release}",
            "model": {
                "file": "weights.pt", "source_checkpoint": name, "sha256": entries[name]["sha256"],
                "architecture": training["cfg"], "parameters": training["params"],
                "streaming_tensor_bytes": measured["streaming_tensor_bytes"],
                "training_targets": training["train_target_bytes"], "seed": training["args"]["seed"],
                ("selection_validation_bpb" if args.release == "v1" else "validation_bpb"): val_bpb,
                "training_record": record_path.relative_to(ROOT).as_posix(),
            },
            "archive": {"file": bundle.name, "sha256": bundle_sha256, "bytes": bundle.stat().st_size},
            "checkpoints": dict(sorted(entries.items())),
        }
        if continuation:
            manifest["model"]["cumulative_training_targets"] = (
                training["train_target_bytes"] + training["initialization"]["training_targets"]
            )
            manifest["archive"]["contents_checksums"] = checksum_path
        if upstream_manifest:
            manifest["upstream_manifest"] = upstream_manifest
        if transfer:
            manifest["transfer_benchmarks"] = transfer
        if training_mixture:
            manifest["training_mixture"] = training_mixture
        (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Packaged {len(entries)} checkpoints in {bundle}")
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
