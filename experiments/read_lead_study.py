"""Apply the reading shared by the test bed's lead studies to their score records.

weight_precision_with_data, cell_dither_with_data and randomness_at_best_rate read the same way: an arm is a fix for
late learning if (a) it leads its base at the long run length by at least --min-lead on the mean of the seeds and at
every seed, and (b) its mean lead at the long length is at least --min-growth larger than at the short length. A lead
by (a) without (b) is a constant gain. cell_shape_with_data reads (a) alone, with --min-wins of the seeds instead of
all. A lead is the base's bits per byte minus the arm's, so a positive lead is a better arm.

usage:
  python experiments/read_lead_study.py --scores FOLDER --base-long bed24-s1 bed24-s2 --base-short bed6-s1 bed6-s2 \
      --arm dither wp-dither-24k-s1 wp-dither-24k-s2 --arm-short dither wp-dither-6k-s1 wp-dither-6k-s2 [--json OUT]
Seeds pair by position: the first name of an arm pairs with the first name of the base.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bpb(folder, name):
    path = Path(folder) / f"{name}.json"
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    return record["checkpoints"][0]["bpb"] if record.get("checkpoints") else record["fast_output_synapses"][0]["bpb"]


def leads(folder, base, arm):
    """Paired leads seed by seed, None where either score is missing."""
    return [None if bpb(folder, b) is None or bpb(folder, a) is None else bpb(folder, b) - bpb(folder, a)
            for b, a in zip(base, arm)]


def read(folder, base_long, base_short, arms_long, arms_short, min_lead, min_growth, min_wins):
    verdicts = {}
    for label, names in arms_long.items():
        long = leads(folder, base_long, names)
        row = {"long_leads": [None if x is None else round(x, 5) for x in long]}
        if any(x is None for x in long):
            row["verdict"] = "incomplete: " + ", ".join(n for n, x in zip(names, long) if x is None) + " not scored"
            verdicts[label] = row
            continue
        mean = sum(long) / len(long)
        wins = sum(x > 0 for x in long)
        needed = len(long) if min_wins is None else min_wins
        a = mean >= min_lead and wins >= needed
        row |= {"long_mean_lead": round(mean, 5), "seeds_ahead": wins, "a_lead": a}
        if label in arms_short:
            short = leads(folder, base_short, arms_short[label])
            row["short_leads"] = [None if x is None else round(x, 5) for x in short]
            if any(x is None for x in short):
                row["verdict"] = "incomplete: the short runs are not all scored"
                verdicts[label] = row
                continue
            growth = mean - sum(short) / len(short)
            b = growth >= min_growth
            row |= {"short_mean_lead": round(sum(short) / len(short), 5), "growth": round(growth, 5), "b_growth": b}
            row["verdict"] = ("pass: a fix for late learning" if a and b else
                              "a constant gain, not a fix for late learning" if a else "fail")
        else:
            row["verdict"] = "pass" if a else "fail"
        verdicts[label] = row
    return verdicts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--base-long", nargs="+", required=True)
    parser.add_argument("--base-short", nargs="+", default=[])
    parser.add_argument("--arm", nargs="+", action="append", default=[], metavar=("LABEL", "NAME"))
    parser.add_argument("--arm-short", nargs="+", action="append", default=[], metavar=("LABEL", "NAME"))
    parser.add_argument("--min-lead", type=float, default=0.010)
    parser.add_argument("--min-growth", type=float, default=0.005)
    parser.add_argument("--min-wins", type=int, help="seeds that must lead; every seed when omitted")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    arms_long = {label: names for label, *names in args.arm}
    arms_short = {label: names for label, *names in args.arm_short}
    for label, names in {**arms_long, **arms_short}.items():
        if len(names) != len(args.base_long):
            parser.error(f"arm {label} names {len(names)} runs; the base names {len(args.base_long)}")
    verdicts = read(args.scores, args.base_long, args.base_short, arms_long, arms_short, args.min_lead,
                    args.min_growth, args.min_wins)
    text = json.dumps(verdicts, indent=1)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
