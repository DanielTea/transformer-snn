"""Regenerate the matched spiking byte model leaderboards and figures."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, RESULTS, TEXT, VIOLET, heading, plt, read_result, save, style


ROOT = RESULTS.parent
LABELS = {
    "always": ("Always-read spiking bank", "Always read"),
    "forecast": ("Forecast-keyed spiking bank", "Forecast key"),
    "context": ("Context tags, fixed thresholds", "Fixed thresholds"),
    "learned": ("Context tags, learned thresholds", "Learned thresholds"),
    "transformer_local": ("Local Transformer", "Local Transformer"),
    "full": ("Full-attention Transformer", "Full attention"),
}
ARMS = {
    "small": ("always", "forecast", "context", "learned", "transformer_local", "full"),
    "large": ("context", "learned", "transformer_local", "full"),
}
COLORS = {
    "always": "#138372", "forecast": "#7a6a50", "context": BLUE, "learned": VIOLET,
    "transformer_local": ORANGE, "full": GRAY,
}
CORPORA = (("enwik8", "enwik8", 4_999_999), ("pg19_full", "16 complete PG-19 books", 8_404_368))
READ_ME_START = "<!-- event-lm-leaderboard:start -->"
READ_ME_END = "<!-- event-lm-leaderboard:end -->"


def summarize(values):
    return {"mean": mean(values), "sd": stdev(values), "by_seed": values}


def read_arm(arm, scale, activity):
    scores = {}
    sources = {}
    metadata = None
    for corpus, _, targets in CORPORA:
        source = f"event_lm/{corpus}_threshold_scale_{arm}_{scale}.json"
        record = read_result(source)
        primary = [row for row in record["models"] if "intervention" not in row]
        if record["target_bytes"] != targets or len(primary) != 3:
            raise ValueError(f"{source} needs three complete test records")
        rows = []
        for row in primary:
            train = read_result(f'event_lm/{row["run_name"]}.json')
            if row["target_bytes"] != targets:
                raise ValueError(f"{source} contains a partial test run")
            rows.append((train["args"]["seed"], train["train_target_bytes"], row))
        rows.sort(key=lambda item: item[0])
        if [seed for seed, _, _ in rows] != [0, 1, 2]:
            raise ValueError(f"{source} needs seeds 0, 1, and 2")
        definitions = {
            (entry["parameters"], entry["streaming_tensor_bytes"], targets_seen) for _, targets_seen, entry in rows
        }
        if len(definitions) != 1:
            raise ValueError(f"{source} mixes parameter, state, or training budgets")
        definition = definitions.pop()
        if metadata is not None and metadata != definition:
            raise ValueError(f"{arm} has inconsistent metadata across corpora")
        metadata = definition
        scores[corpus] = summarize([row["bpb"] for _, _, row in rows])
        sources[corpus] = source
    parameters, state, training_targets = metadata
    pulses = [activity[Path(row["checkpoint"]).name] for _, _, row in rows] if arm not in ("transformer_local", "full") else []
    return {
        "arm": arm, "model": LABELS[arm][0], "parameters": parameters, "state_bytes": state,
        "training_targets": training_targets, "scores": scores, "score_files": sources,
        "pulses_per_byte": summarize(pulses) if pulses else None,
    }


def build():
    leaderboard = {"unit": "bits per byte", "seeds": [0, 1, 2], "groups": {}}
    for scale, arms in ARMS.items():
        audit = read_result(f"event_lm/threshold_scale_{scale}_activity.json")
        activity = {Path(row["checkpoint"]).name: row["total_pulses_per_byte"] for row in audit["models"]}
        rows = [read_arm(arm, scale, activity) for arm in arms]
        exposures = {row["training_targets"] for row in rows}
        if len(exposures) != 1:
            raise ValueError(f"{scale} contains unequal training exposure")
        for corpus, _, _ in CORPORA:
            ordered = sorted(rows, key=lambda row: row["scores"][corpus]["mean"])
            for rank, row in enumerate(ordered, start=1):
                row["scores"][corpus]["rank"] = rank
        leaderboard["groups"][scale] = rows
    return leaderboard


def score_cell(stats):
    score = f'{stats["mean"]:.5f}'
    return f'**{score}**' if stats["rank"] == 1 else score


def state_size(state):
    return f"{state / 1024 ** 2:g} MiB" if state >= 1024 ** 2 else f"{state:,} B"


def markdown_table(rows, *, include_ranks):
    if include_ranks:
        lines = [
            "| Model | Parameters | State | enwik8 | Rank | PG-19 | Rank | Pulses/byte |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        lines = ["| Model | enwik8 | PG-19 | State |", "|---|---:|---:|---:|"]
    for row in rows:
        enwik8, pg19 = (row["scores"][corpus] for corpus, _, _ in CORPORA)
        state = state_size(row["state_bytes"])
        if include_ranks:
            pulses = row["pulses_per_byte"]
            pulse_text = f'{pulses["mean"]:,.0f}' if pulses else "—"
            lines.append(f'| {row["model"]} | {row["parameters"]:,} | {state} | {score_cell(enwik8)} | '
                         f'{enwik8["rank"]} | {score_cell(pg19)} | {pg19["rank"]} | {pulse_text} |')
        else:
            lines.append(f'| {row["model"]} | {score_cell(enwik8)} | {score_cell(pg19)} | {state} |')
    return "\n".join(lines)


def write_leaderboards(data):
    paragraphs = [
        "# Spiking byte model leaderboards",
        "Scores are three-seed means in bits per byte. Lower scores rank higher. "
        "Every row at a size uses the same enwik8 training windows and scores "
        "4,999,999 enwik8 targets and 8,404,368 targets from 16 complete PG-19 books.",
        "State is persistent tensor storage for one stream. The separate full-attention "
        "reference attends to a causal prefix of up to 512 bytes and reports its FP32 key "
        "and value storage at that length. Pulse counts use a native 4,096-byte enwik8 test sample.",
    ]
    for scale, rows in data["groups"].items():
        paragraphs.extend([
            f'## {scale.capitalize()} comparison · {rows[0]["training_targets"]:,} training targets per seed',
            markdown_table(rows, include_ranks=True),
        ])
    paragraphs.extend([
        "![Held-out loss by model size and corpus](../results/figures/event_lm_loss.png)",
        "![Loss versus native spike activity for the spiking banks](../results/figures/event_lm_pulse_tradeoff.png)",
        "[Per-seed scores and source paths](../results/event_lm/threshold_scale_leaderboard.json) · "
        "[Interventions and research notes](SPIKE_CODE_FORECAST.md#learned-context-keys-and-firing-thresholds) · "
        "[Model and reproduction guide](../models/spiking-byte-lm/README.md)",
        "Regenerate this page and its figures with `python experiments/plot_event_lm_study.py`.",
    ])
    (ROOT / "docs/LEADERBOARD.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
    readme = ROOT / "README.md"
    before, marker, remainder = readme.read_text(encoding="utf-8").partition(READ_ME_START)
    _, end, after = remainder.partition(READ_ME_END)
    if not marker or not end:
        raise ValueError("README.md needs both event model leaderboard markers")
    table = markdown_table(data["groups"]["large"], include_ranks=False)
    readme.write_text(f"{before}{marker}\n\n{table}\n\n{end}{after}", encoding="utf-8")


def plot_loss(data, output):
    fig, axes = plt.subplots(2, 2, sharey="row", figsize=(11.8, 7.4))
    fig.subplots_adjust(left=0.194, right=0.97, bottom=0.112, top=0.809, hspace=0.49, wspace=0.20)
    heading(fig, "Held-out next-byte loss",
            "Three matched seeds per model · enwik8 and 16 complete PG-19 books")
    for index, rows in enumerate(data["groups"].values()):
        params = f'{rows[0]["parameters"] / 1e3:.0f}k'
        for col, (corpus, name, _) in enumerate(CORPORA):
            ax = axes[index, col]
            style(ax, grid_axis="x")
            ax.set_title(f"About {params} parameters · {name}", fontsize=11, loc="left", color=TEXT)
            for y, row in enumerate(rows):
                metric = row["scores"][corpus]
                color = COLORS[row["arm"]]
                values = metric["by_seed"]
                ax.hlines(y, min(values), max(values), color=color, linewidth=1.8, alpha=0.55)
                ax.scatter(values, [y] * len(values), color=color, s=11, alpha=0.65, zorder=3)
                ax.scatter(metric["mean"], y, color=color, s=41, edgecolors="white", linewidths=0.9, zorder=4)
            ax.set_xlabel("Bits per byte", color=MUTED, labelpad=7)
            ax.set_yticks(range(len(rows)), [LABELS[row["arm"]][1] for row in rows])
            ax.set_ylim(len(rows) - 0.45, -0.55)
            ax.margins(x=0.10)
    fig.text(0.035, 0.033, "Large points are means. Lines span the seeds. PG-19 scores use validation temperatures.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_loss.png")


def plot_pulses(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    fig.subplots_adjust(left=0.083, right=0.973, bottom=0.24, top=0.745, wspace=0.28)
    heading(fig, "Loss and spike activity",
            "Spiking banks · count audit on 4,096 held-out enwik8 bytes · three-seed means")
    for ax, (corpus, name, _) in zip(axes, CORPORA, strict=True):
        style(ax)
        ax.set_title(name, color=TEXT, size=11, loc="left")
        ax.set_xlabel("Counted pulses per byte, thousands", color=MUTED, labelpad=8)
        ax.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
        ax.axvline(310, color=GRAY, linestyle="--", linewidth=1.2)
        for scale, rows in data["groups"].items():
            for row in rows:
                if row["pulses_per_byte"] is None:
                    continue
                activity = row["pulses_per_byte"]["mean"] / 1000
                loss = row["scores"][corpus]["mean"]
                ax.scatter(activity, loss, color=COLORS[row["arm"]], marker="o" if scale == "small" else "D",
                           s=59, edgecolors="white", linewidths=0.8, zorder=3)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[arm], label=LABELS[arm][1])
               for arm in ARMS["small"] if arm not in ("transformer_local", "full")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, frameon=False,
               labelcolor=TEXT, fontsize=9)
    fig.text(0.035, 0.032, "Circles mark small models. Diamonds mark large models. The dashed line marks the training budget.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_pulse_tradeoff.png")


def main():
    data = build()
    summary = RESULTS / "event_lm/threshold_scale_leaderboard.json"
    summary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    write_leaderboards(data)
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plot_loss(data, output)
    plot_pulses(data, output)


if __name__ == "__main__":
    main()
