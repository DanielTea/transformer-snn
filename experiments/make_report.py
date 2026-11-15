"""Build the self-contained HTML research report from results/*.json.

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
        if r["args"].get("tag", "") != tag:
            continue
        if r["args"].get("train_frac", 1.0) != 1.0:
            continue
        g[r["args"]["variant"]].append(r)
    return g


def agg(rs, key, scale=1.0):
    xs = [r[key] * scale for r in rs]
    return {"mean": st.mean(xs), "sd": st.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs), "all": xs}


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
                           "energy_uJ": e["energy_mJ_per_sample"] * 1e3, "fr": e["firing_rate"],
                           "nonlin": e["nonlinear_per_sample"], "params": rs[0]["params"]})
    for v, rs in load("vision").items():
        e = rs[0]["energy_per_image"]
        data["vision"].append({"v": v, "label": SHORT.get(v, v), "role": ROLE.get(v, "ref"),
                               **agg(rs, "test_acc", 100), "energy_uJ": e["energy_mJ_per_sample"] * 1e3,
                               "fr": e["firing_rate"], "params": rs[0]["params"],
                               "curve": [p["test_acc"] * 100 for p in rs[0]["curve"]]})
    return data


TEMPLATE = (Path(__file__).with_name("report_template.html")).read_text


def main(out):
    data = build_data()
    page = TEMPLATE().replace("/*__DATA__*/null", json.dumps(data))
    # prose written after reading the results: results/report_text.json {"VERDICT": "...", ...}
    tf = RESULTS / "report_text.json"
    texts = json.loads(tf.read_text()) if tf.exists() else {}
    for key in ("VERDICT", "RECALL_TEXT", "LM_TEXT", "VISION_TEXT", "SAMPLE_TEXT"):
        page = page.replace("{{%s}}" % key, html.escape(texts.get(key, "(pending)")))
    Path(out).write_text(page)
    print("wrote", out, {k: len(v) for k, v in data.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report.html")
