"""Static figures for the README (results/figures/*.png)."""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from summarize import LABELS  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIG = RESULTS / "figures"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
CATEGORY = {  # colour follows the role of the variant, not its rank
    "ours": BLUE, "ours_k4": BLUE, "csa": BLUE, "ours_nocomp": BLUE, "ours_nonorm": BLUE, "ours_binqk": BLUE, "ours_nopbn": BLUE,
    "spikformer": ORANGE, "ssa_norm": ORANGE, "sdt": ORANGE, "wta": ORANGE, "wta3": ORANGE,
    "spikformer_cdglu": ORANGE, "spikformer_nopbn": ORANGE,
    "ann": AQUA, "spk_softmax": AQUA,
}
LEGEND = [(BLUE, "this work (spike-driven)"), (ORANGE, "prior spike-driven attention"),
          (AQUA, "reference (not spike-driven)")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _load(task):
    g = defaultdict(list)
    for f in (RESULTS / task).glob("*.json"):
        r = json.loads(f.read_text())
        if r["args"].get("tag"):
            continue
        frac = r["args"].get("train_frac", 1.0)
        g[(r["args"]["variant"], frac)].append(r)
    return g


def bar_chart(task, metric, xlabel, fname, scale=100.0, order=None, title=""):
    g = _load(task)
    rows = []
    for (v, frac), rs in g.items():
        if frac != 1.0 or (order and v not in order):
            continue
        xs = [r[metric] * scale for r in rs]
        rows.append((v, st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0, len(xs)))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(rows) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ys = range(len(rows))
    for y, (v, m, s, n) in zip(ys, rows):
        ax.barh(y, m, height=0.62, color=CATEGORY.get(v, INK2), xerr=s if s else None,
                error_kw=dict(ecolor=INK2, lw=1, capsize=2))
        ax.text(m + (s or 0) + 0.8, y, f"{m:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(list(ys), [LABELS.get(r[0], r[0]).replace("**", "") for r in rows], fontsize=8.5, color=INK)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_xlim(0, max(r[1] + r[2] for r in rows) * 1.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c, _ in LEGEND]
    ax.legend(handles, [l for _, l in LEGEND], loc="lower right", fontsize=7.5, frameon=False, labelcolor=INK2)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / fname, facecolor=SURFACE)
    plt.close(fig)


def lm_curves(fname="lm_curves.png", show=("ann", "spk_softmax", "ours", "spikformer", "wta", "sdt")):
    g = _load("lm")
    if not g:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    dashes = {"ann": (None, None), "spk_softmax": (4, 2), "ours": (None, None), "csa": (4, 2),
              "spikformer": (None, None), "wta": (4, 2), "sdt": (1, 2), "ssa_norm": (6, 2)}
    for v in show:
        rs = g.get((v, 1.0))
        if not rs:
            continue
        c = rs[0]["curve"]
        xs = [p["tokens"] / 1e6 for p in c]
        ys = [st.mean(r["curve"][i]["val_bpb"] for r in rs) for i in range(len(c))]
        line, = ax.plot(xs, ys, color=CATEGORY.get(v, INK2), lw=2, label=LABELS.get(v, v).replace("**", ""))
        if dashes.get(v, (None, None))[0]:
            line.set_dashes(dashes[v])
        ax.text(xs[-1] * 1.01, ys[-1], f"{ys[-1]:.2f}", fontsize=8, color=INK, va="center")
    ax.set_xlabel("training tokens (millions)", color=INK2, fontsize=9)
    ax.set_ylabel("validation bits per byte (lower is better)", color=INK2, fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2)
    lo = min(min(p["val_bpb"] for p in r["curve"]) for rs in g.values() for r in rs)
    ax.set_ylim(lo - 0.1, min(4.0, ax.get_ylim()[1]))
    ax.set_title("enwik8 byte-level LM: learning curves (sample efficiency)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / fname, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    bar_chart("recall", "final_acc", "associative-recall accuracy (%), mean ± sd over 3 seeds", "recall.png",
              title="Multi-query associative recall (8 pairs) — can the attention select?")
    lm_curves()
    bar_chart("vision", "test_acc", "CIFAR-10 test accuracy (%)", "cifar10.png",
              title="CIFAR-10, small spiking ViT (same backbone, same budget)")
    print("figures in", FIG)
