"""How far a decoder's state reads have moved their addresses, at one or more checkpoints of a run.

usage: read_addresses.py UPDATES=CHECKPOINT... [--averaged] [--json OUT]

A read head addresses a cell by the match of its probe with the cell's tag, times a gain, plus an offset for
the cell's age in bytes and, where the read has a clock, an offset for its age in beats. Offsets and gains are
charges: one unit multiplies a cell's rate in the race by e. For every head this prints ``near``, the mean
offset of the eight newest ages in bytes over the mean of the 64 oldest, the range of its offsets by age in
bytes, its gain, its offsets by age in beats and, where beats mark their cells, the further offsets of the
marked cells. The values are the stored ones times the read's address rate. Only the checkpoint files are
read, so no model is built.
"""
import argparse
import json
from pathlib import Path

import torch


def addresses(path, averaged):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    if averaged and checkpoint.get("averaged_parameters"):
        state = {**state, **checkpoint["averaged_parameters"]}
    rate = checkpoint["cfg"].get("read_address_rate", 1.0)
    rows = []
    for index in checkpoint["cfg"].get("read_after") or ():
        name = f"state_reads.{index}."
        age = state[name + "age"].float() * rate
        gain = state.get(name + "gain")
        beats = state.get(name + "beat_age")
        marks = state.get(name + "beat_mark")
        for head in range(age.shape[0]):
            row = {"read_after": int(index), "head": head,
                   "near": float(age[head, :8].mean() - age[head, -64:].mean()),
                   "range": float(age[head].max() - age[head].min())}
            if gain is not None:
                row["gain"] = float(gain[head].flatten()[0] * rate)
            if beats is not None:
                row["by_age_in_beats"] = (beats[head].float() * rate).tolist()
            if marks is not None:
                row["marked_by_age_in_beats"] = (marks[head].float() * rate).tolist()
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+", metavar="UPDATES=CHECKPOINT")
    parser.add_argument("--averaged", action="store_true",
                        help="read the running weight average where a checkpoint stores one")
    parser.add_argument("--json", type=Path, help="also write the rows to this file")
    args = parser.parse_args()
    record = {"averaged": args.averaged, "checkpoints": []}
    for item in args.checkpoints:
        updates, _, path = item.partition("=")
        rows = addresses(Path(path), args.averaged)
        record["checkpoints"].append({"updates": int(updates), "heads": rows})
        print(f"{int(updates):>6} updates")
        for row in rows:
            line = (f"  read after block {row['read_after']} head {row['head']}: near {row['near']:+.3f}  "
                    f"range {row['range']:.3f}")
            if "gain" in row:
                line += f"  gain {row['gain']:+.3f}"
            if "by_age_in_beats" in row:
                line += "  by age in beats " + " ".join(f"{value:+.2f}" for value in row["by_age_in_beats"])
            if "marked_by_age_in_beats" in row:
                line += "  marked " + " ".join(f"{value:+.2f}" for value in row["marked_by_age_in_beats"])
            print(line)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
