"""The learned age offsets of a stack's state reads (run from the repository's root):

    python experiments/read_age_offsets.py OUT.json NAME=CHECKPOINT [NAME=CHECKPOINT ...]

A state read adds to every cell's charge in its race an offset for the cell's age in bytes, one learned offset per
head and age within the read's window, stored at the read address rate (snnformer/event_lm.py, _StateRead.aged).
For every checkpoint: the mean offset times the address rate over all reads and heads in bands of age, the band
means of every read, and the first age from which the mean over all reads and heads stays below zero for five ages
in a row. No stream is run; the numbers are the checkpoint's own."""
import json
import sys
from pathlib import Path

import torch

BANDS = [(0, 25), (25, 50), (50, 75), (75, 100), (100, 125), (125, 140), (140, 150), (150, 175), (175, 200),
         (200, 256)]


def offsets(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights, rate = checkpoint["state_dict"], checkpoint["cfg"]["read_address_rate"]
    keys = sorted(k for k in weights if k.endswith(".age"))
    return keys, torch.stack([weights[k].float() * rate for k in keys])  # [reads, heads, ages]


def bands(ages: torch.Tensor) -> dict:
    return {f"{lo}-{hi}": round(ages[..., lo:min(hi, ages.shape[-1])].mean().item(), 2)
            for lo, hi in BANDS if lo < ages.shape[-1]}


def main():
    out, found = Path(sys.argv[1]), {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        keys, ages = offsets(Path(path))
        mean = ages.mean(dim=(0, 1))
        below = next((a for a in range(len(mean) - 4) if (mean[a:a + 5] < 0).all()), None)
        found[name] = {"reads": len(keys), "heads": ages.shape[1], "ages": ages.shape[2],
                       "mean over reads and heads, by age band (bytes)": bands(ages),
                       "by read": {k.rsplit(".", 1)[0]: bands(ages[i]) for i, k in enumerate(keys)},
                       "first age of five in a row below zero (mean over reads and heads)": below}
        print(name, below, bands(ages), flush=True)
    out.write_text(json.dumps(found, indent=1) + "\n")


if __name__ == "__main__":
    main()
