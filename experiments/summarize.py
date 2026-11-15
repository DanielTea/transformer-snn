"""Aggregate results/*/*.json into markdown tables (mean ± std over seeds)."""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

LABELS = {
    "ann": "ANN Transformer (GELU, softmax) — reference",
    "spk_softmax": "Spiking + softmax attention (not spike-driven)",
    "spikformer": "Spikformer SSA (≡ Meta-SpikeFormer SDSA-3)",
    "ssa_norm": "SSA + 1/#keys normalisation (extra baseline)",
    "sdt": "Spike-driven Transformer SDSA",
    "wta": "WTA Spiking Transformer, hard WTA (Zhou+ 2026)",
    "wta3": "WTA Spiking Transformer, top-3 (Zhou+ 2026)",
    "csa": "CSA attention + standard MLP (ours)",
    "ours": "**CSA + CD-GLU (ours, full)**",
    "ours_k4": "**CSA-4 (graded attention spikes) + CD-GLU (ours)**",
    "ours_nocomp": "ours − competition (soft scores)",
    "ours_nonorm": "ours − shunting normalisation",
    "spikformer_cdglu": "Spikformer SSA + CD-GLU",
    "ours_binqk": "ours with binary Q/K",
    "ours_nopbn": "ours − pre-neuron BN on residual",
    "spikformer_nopbn": "Spikformer SSA, original backbone (no pre-BN)",
}
ORDER = list(LABELS)


def fmt(xs, scale=1.0, digits=2):
    xs = [x * scale for x in xs]
    if len(xs) == 1:
        return f"{xs[0]:.{digits}f}"
    return f"{st.mean(xs):.{digits}f} ± {st.stdev(xs):.{digits}f}"


def load(task):
    groups = defaultdict(list)
    for f in sorted((RESULTS / task).glob("*.json")):
        r = json.loads(f.read_text())
        a = r["args"]
        key = (a["variant"], r["cfg"]["neuron"], r["cfg"]["T"], a.get("tag", ""), a.get("pairs"),
               a.get("train_frac"))
        groups[key].append(r)
    return groups


def table(task):
    groups = load(task)
    if not groups:
        return ""
    rows = []
    if task == "recall":
        rows.append("| variant | steps | params | accuracy (%) | per seed | seeds solved (>90%) |")
        rows.append("|---|---|---|---|---|---|")
        for key in sorted(groups, key=lambda k: (k[3], ORDER.index(k[0]) if k[0] in ORDER else 99)):
            rs = groups[key]
            accs = [r["final_acc"] * 100 for r in rs]
            rows.append(f"| {LABELS.get(key[0], key[0])} | {rs[0]['args']['steps']} | {rs[0]['params']:,} | "
                        f"{fmt([a / 100 for a in accs], 100, 1)} | {' / '.join(f'{a:.1f}' for a in accs)} | "
                        f"{sum(a > 90 for a in accs)}/{len(accs)} |")
    elif task == "lm":
        rows.append("| variant | neuron/T | params | val bpb | test bpb | firing rate | est. energy / token (µJ) | seeds |")
        rows.append("|---|---|---|---|---|---|---|---|")
        for key in sorted(groups, key=lambda k: ORDER.index(k[0]) if k[0] in ORDER else 99):
            rs = groups[key]
            e = rs[0]["energy_per_token"]
            rows.append(f"| {LABELS.get(key[0], key[0])} | {key[1]}/T{key[2]} | {rs[0]['params']:,} | "
                        f"{fmt([r['val_bpb'] for r in rs], 1, 3)} | {fmt([r['test_bpb'] for r in rs], 1, 3)} | "
                        f"{e['firing_rate']:.3f} | {e['energy_mJ_per_sample']*1e3:.2f} | {len(rs)} |")
    elif task == "vision":
        rows.append("| variant | neuron/T | train frac | params | test acc (%) | firing rate | est. energy / image (µJ) | seeds |")
        rows.append("|---|---|---|---|---|---|---|---|")
        for key in sorted(groups, key=lambda k: (-(k[5] or 1), ORDER.index(k[0]) if k[0] in ORDER else 99)):
            rs = groups[key]
            e = rs[0]["energy_per_image"]
            rows.append(f"| {LABELS.get(key[0], key[0])} | {key[1]}/T{key[2]} | {key[5]} | {rs[0]['params']:,} | "
                        f"{fmt([r['test_acc'] for r in rs], 100, 2)} | {e['firing_rate']:.3f} | "
                        f"{e['energy_mJ_per_sample']*1e3:.1f} | {len(rs)} |")
    return "\n".join(rows)


if __name__ == "__main__":
    for task in sys.argv[1:] or ["recall", "lm", "vision"]:
        print(f"\n### {task}\n")
        print(table(task))
