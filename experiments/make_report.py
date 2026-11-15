"""Build the HTML research report from results/*.json.

usage: python make_report.py <out.html>
"""

from __future__ import annotations

import html
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

from summarize import LABELS

RESULTS = Path(__file__).resolve().parents[1] / "results"

ROLE = {
    "ours": "ours", "ours_k4": "ours", "csa": "ours", "ours_nocomp": "ours", "ours_nonorm": "ours", "ours_binqk": "ours",
    "ours_nopbn": "ours",
    "spikformer": "prior", "ssa_norm": "prior", "sdt": "prior", "wta": "prior", "wta3": "prior",
    "spikformer_cdglu": "prior", "spikformer_nopbn": "prior",
    "ann": "ref", "spk_softmax": "ref",
}
SHORT = {
    "ours": "ShuntFormer (CSA + CD-GLU)", "ours_k4": "ShuntFormer-4 (graded CSA)", "csa": "CSA + plain MLP", "ours_nocomp": "− competition",
    "ours_nonorm": "− shunting normalisation", "ours_binqk": "binary Q/K", "ours_nopbn": "− residual pre-BN",
    "spikformer": "Spikformer SSA", "ssa_norm": "SSA ÷ #keys", "sdt": "Spike-driven SDSA",
    "wta": "WTA (hard, 2026)", "wta3": "WTA (top-3, 2026)", "spikformer_cdglu": "SSA + CD-GLU",
    "spikformer_nopbn": "SSA, original backbone", "ann": "ANN transformer", "spk_softmax": "softmax on spikes",
}


def load(task, tag=""):
    g = defaultdict(list)
    for f in sorted((RESULTS / task).glob("*.json")):
        r = json.loads(f.read_text())
        args = r.get("args")
        if args is None or args.get("tag", "") != tag:
            continue
        if args.get("train_frac", 1.0) != 1.0:
            continue
        g[args["variant"]].append(r)
    return g


def agg(rs, key, scale=1.0):
    xs = [r[key] * scale for r in rs]
    return {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs), "all": xs}


def dendrite_summary():
    recall = load("recall_distance", "gap_v1")
    event, control = recall["dendrite_event"], recall["transformer"]
    max_gap = str(max(map(int, event[0]["final_acc_by_gap"])))
    anchored = json.loads((RESULTS / "recall_distance/dendrite_event_s0_anchored_extrapolation.json").read_text())
    anchored_gap = max(map(int, anchored["accuracy_by_gap"]))
    anchored_total = anchored["samples_per_gap"] * anchored["queries"]

    successor = RESULTS / "lm_successor"
    calibrated = json.loads((successor / "scaled_v2_continuous.json").read_text())["results"]
    trained = json.loads((successor / "end_to_end_continuous.json").read_text())["results"]
    enwik8, trained_enwik8 = (
        next(row for row in rows if row["source"] == "enwik8 test")
        for rows in (calibrated, trained)
    )
    books = [row for row in calibrated if row["source"] != "enwik8 test"]
    trained_books = [row for row in trained if row["source"] != "enwik8 test"]

    def byte_weighted(rows, key):
        return sum(row[key] * row["target_bytes"] for row in rows) / sum(row["target_bytes"] for row in rows)

    return {
        "DENDRITE_TRAIN_GAP": str(event[0]["args"]["train_max_gap"]),
        "DENDRITE_TEST_GAP": max_gap,
        "DENDRITE_EVENT": f'{100 * st.mean(r["final_acc_by_gap"][max_gap] for r in event):.2f}%',
        "DENDRITE_CONTROL": f'{100 * st.mean(r["final_acc_by_gap"][max_gap] for r in control):.2f}%',
        "DENDRITE_SEEDS": str(len(event)),
        "DENDRITE_ANCHORED_GAP": f"{anchored_gap:,}",
        "DENDRITE_ANCHORED_CORRECT": f'{anchored_total * anchored["accuracy_by_gap"][str(anchored_gap)]:.0f}',
        "DENDRITE_ANCHORED_TOTAL": str(anchored_total),
        "DENDRITE_ENWIK8_COUNT": f'{enwik8["target_bytes"]:,}',
        "DENDRITE_ENWIK8_BASE": f'{enwik8["base_bpb"]:.3f}',
        "DENDRITE_ENWIK8_CALIBRATED": f'{enwik8["spike_bpb"]:.3f}',
        "DENDRITE_ENWIK8_TRAINED": f'{trained_enwik8["spike_bpb"]:.3f}',
        "DENDRITE_PG19_BOOKS": str(len(books)),
        "DENDRITE_PG19_COUNT": f'{sum(row["target_bytes"] for row in books):,}',
        "DENDRITE_PG19_BASE": f'{byte_weighted(books, "base_bpb"):.3f}',
        "DENDRITE_PG19_CALIBRATED": f'{byte_weighted(books, "spike_bpb"):.3f}',
        "DENDRITE_PG19_TRAINED": f'{byte_weighted(trained_books, "spike_bpb"):.3f}',
    }


def build_data():
    data = {"recall": [], "recall_long": [], "lm": [], "vision": []}
    for v, rs in load("recall", "long").items():
        a = agg(rs, "final_acc", 100)
        data["recall_long"].append({"v": v, "label": SHORT.get(v, v), "role": ROLE.get(v, "ref"), **a,
                                    "solved": sum(x > 90 for x in a["all"])})
    for v, rs in load("recall").items():
        a = agg(rs, "final_acc", 100)
        data["recall"].append({"v": v, "label": SHORT.get(v, v), "role": ROLE.get(v, "ref"), **a})
    for v, rs in load("lm").items():
        c = rs[0]["curve"]
        curve = [[p["tokens"] / 1e6, st.mean(r["curve"][i]["val_bpb"] for r in rs)] for i, p in enumerate(c)]
        e = rs[0]["energy_per_token"]
        data["lm"].append({"v": v, "label": SHORT.get(v, v), "role": ROLE.get(v, "ref"),
                           "val": agg(rs, "val_bpb"), "test": agg(rs, "test_bpb"), "curve": curve,
                           "energy_uJ": e["energy_mJ_per_sample"] * 1e3 if e else None,
                           "fr": e["firing_rate"] if e else None,
                           "nonlin": e["nonlinear_per_sample"] if e else None, "params": rs[0]["params"]})
    for v, rs in load("vision").items():
        e = rs[0]["energy_per_image"]
        data["vision"].append({"v": v, "label": SHORT.get(v, v), "role": ROLE.get(v, "ref"),
                               **agg(rs, "test_acc", 100), "energy_uJ": e["energy_mJ_per_sample"] * 1e3,
                               "fr": e["firing_rate"], "params": rs[0]["params"],
                               "curve": [p["test_acc"] * 100 for p in rs[0]["curve"]]})
    return data


def spiking_byte_tables():
    path = RESULTS / "event_lm" / "threshold_scale_leaderboard.json"
    study = json.loads(path.read_text(encoding="utf-8"))
    sections = []
    for scale, rows in study["groups"].items():
        exposure = rows[0]["training_targets"]
        sections.append(f"<h3>{scale.capitalize()} comparison · {exposure:,} training targets per seed</h3>")
        sections.append('<div class="tablewrap"><table><thead><tr><th scope="col">Decoder</th>'
                        '<th scope="col" class="num">Parameters</th><th scope="col" class="num">State</th>'
                        '<th scope="col" class="num">enwik8</th><th scope="col" class="num">PG-19</th>'
                        '</tr></thead><tbody>')
        for row in rows:
            cells = [f'{row["parameters"]:,}', f'{row["state_bytes"]:,} B']
            for corpus in ("enwik8", "pg19_full"):
                result = row["scores"][corpus]
                score = f'{result["mean"]:.5f}'
                cells.append(f"<strong>{score}</strong>" if result["rank"] == 1 else score)
            body = "".join(f'<td class="num">{value}</td>' for value in cells)
            sections.append(f'<tr><th scope="row">{html.escape(row["model"])}</th>{body}</tr>')
        sections.append("</tbody></table></div>")
    return "\n".join(sections)


TEMPLATE = (Path(__file__).with_name("report_template.html")).read_text


def main(out):
    data = build_data()
    page = TEMPLATE().replace("/*__DATA__*/null", json.dumps(data))
    page = page.replace("<!-- EVENT_LM_TABLES -->", spiking_byte_tables())
    for key, value in dendrite_summary().items():
        page = page.replace("{{%s}}" % key, html.escape(value))
    # prose written after reading the results: results/report_text.json {"VERDICT": "...", ...}
    tf = RESULTS / "report_text.json"
    texts = json.loads(tf.read_text()) if tf.exists() else {}
    for key in ("VERDICT", "RECALL_TEXT", "LM_TEXT", "VISION_TEXT", "SAMPLE_TEXT"):
        page = page.replace("{{%s}}" % key, html.escape(texts.get(key, "(pending)")))
    Path(out).write_text(page)
    print("wrote", out, {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report.html")
