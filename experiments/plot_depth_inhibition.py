"""Plot validation learning curves and native residual pulse activity."""

import argparse

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, RESULTS, TEXT, VIOLET, heading, plt, read_result, save, style


STUDY = "event_lm/depth_inhibition"
SCALE = "event_lm/scale_1630k"
MODELS = (
    ("3 blocks, fixed", f"{SCALE}/event_correction_code_s0_scale_1630k", "wide_fixed", GRAY, "--"),
    ("3 blocks, inhibition", f"{STUDY}/event_correction_code_s0_wide_inhibited", "wide_inhibited", BLUE, "--"),
    ("5 blocks, fixed", f"{STUDY}/event_correction_code_s0_deep_fixed", "deep_fixed", TEXT, "-"),
    ("5 blocks, inhibition", f"{STUDY}/event_correction_code_s0_deep_inhibited", "deep_inhibited", BLUE, "-"),
    ("Full-attention Transformer", f"{SCALE}/transformer_s0_scale_1630k", None, ORANGE, ":"),
)
TRACE = ("5 blocks, inhibition + trace", f"{STUDY}/event_correction_code_s0_deep_trace", "deep_trace", VIOLET, "-")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-trace", action="store_true", help="include the completed trace run and its audit")
    args = parser.parse_args()
    fig, (loss, activity) = plt.subplots(1, 2, figsize=(11.5, 4.8), width_ratios=(1.35, 1))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.74, bottom=0.28, wspace=0.33)
    heading(fig, "Learning curves and residual spike counts",
            "1.63M parameters · seed 0 · 26.2M enwik8 training targets")

    trace_ready = all((RESULTS / path).is_file() for path in
                      (f"{TRACE[1]}.json", f"{STUDY}/native_{TRACE[2]}.json"))
    models = (*MODELS, TRACE) if args.include_trace or trace_ready else MODELS
    for label, record, audit, color, line in models:
        run = read_result(f"{record}.json")
        curve = run["curve"]
        loss.plot([item["tokens"] / 1_000_000 for item in curve], [item["val_bpb"] for item in curve],
                  line, color=color, linewidth=1.9, label=label)
        if audit is not None:
            profile = read_result(f"{STUDY}/native_{audit}.json")["models"][0]["activity_profile"]
            medians = [profile["input"]["p05_p50_p95"][1]]
            medians.extend(profile[f"block_{block}_residual"]["p05_p50_p95"][1]
                           for block in range(run["args"]["depth"]))
            activity.plot(range(len(medians)), medians, line, color=color, linewidth=1.9, marker="o", markersize=4)

    for ax in (loss, activity):
        style(ax)
    loss.set_xlabel("Training targets (millions)", color=MUTED, labelpad=7)
    loss.set_ylabel("Validation bits per byte", color=MUTED, labelpad=9)
    activity.set_xlabel("Residual stream after each block", color=MUTED, labelpad=7)
    activity.set_ylabel("Median absolute pulses per channel", color=MUTED, labelpad=9)
    activity.set_xticks(range(6), ["Input", "1", "2", "3", "4", "5"])
    handles, labels = loss.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.093), ncol=3,
               frameon=False, labelcolor=TEXT, fontsize=9)
    fig.text(0.036, 0.046, "Native activity uses the first 4,096 enwik8 test bytes.", color=MUTED, size=9)
    output = RESULTS / "figures"
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output, "depth_inhibition.png")


if __name__ == "__main__":
    main()
