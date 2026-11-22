"""Score the surprise that telemetry_pilot.py kept against simple baselines, on the datasets' own anomaly labels.

Every method gives every sensor a score for every sample:

* a ``Volley`` run: the bits that the stack spent on the sample;
* ``zscore``: how many standard deviations the value lies from the mean of the ``--window`` samples before it;
* ``ewma``: the squared residual of a forecast by an exponentially weighted mean of the values before it;
* ``ar``: the squared residual of a one-step forecast by an autoregression of order ``--order``, fitted by least
  squares on the run's calibration samples.

The scores of a sensor are made comparable with the other sensors by the median and the median absolute deviation
of its own calibration samples, averaged over the sensors of the run, and smoothed by an exponential mean of span
``--span``. All methods get exactly the same treatment.

Only samples after the calibration stretch are scored. ``auc`` is the area under the ROC curve of the smoothed score
against the labels, which needs no threshold. Every method then sets its own threshold at the 99th percentile of its
score on the calibration samples after the first ``--warm``, so that it raises an alarm about once in a hundred
normal samples, and is measured by its pointwise precision, recall and F1, by the anomalies it catches (an anomaly
is a run of labelled samples: caught when an alarm falls inside it, after a delay counted from its first sample) and
by its false alarms: the alarms that start among normal samples, per thousand of them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BASELINES = ("zscore", "ewma", "ar")


def rolling_zscore(values, window):
    """|x - mean| / std of the ``window`` samples before every sample, as [sensors, samples]."""
    values = values.T
    out = np.zeros_like(values)
    cumulative = np.cumsum(np.pad(values, ((0, 0), (1, 0))), 1)
    squares = np.cumsum(np.pad(values ** 2, ((0, 0), (1, 0))), 1)
    for t in range(1, values.shape[1]):
        start = max(0, t - window)
        count = t - start
        mean = (cumulative[:, t] - cumulative[:, start]) / count
        var = np.maximum((squares[:, t] - squares[:, start]) / count - mean ** 2, 0)
        out[:, t] = np.abs(values[:, t] - mean) / (np.sqrt(var) + 1e-9)
    return out


def ewma_residuals(values, alpha=0.1):
    """The squared residual of forecasting every sample by the exponential mean of the samples before it."""
    values = values.T
    forecast = values[:, 0].copy()
    out = np.zeros_like(values)
    for t in range(1, values.shape[1]):
        out[:, t] = (values[:, t] - forecast) ** 2
        forecast = alpha * values[:, t] + (1 - alpha) * forecast
    return out


def ar_residuals(values, order, calibration):
    """The squared residual of a one-step autoregressive forecast of every sensor, fitted on the calibration."""
    out = np.zeros((values.shape[1], values.shape[0]))
    for sensor in range(values.shape[1]):
        series = values[:, sensor]
        lags = np.stack([series[order - k - 1:len(series) - k - 1] for k in range(order)], 1)
        design = np.concatenate([lags, np.ones((len(lags), 1))], 1)
        target = series[order:]
        fit = slice(0, max(order + 1, calibration - order))
        coef, *_ = np.linalg.lstsq(design[fit], target[fit], rcond=None)
        out[sensor, order:] = (target - design @ coef) ** 2
    return out


def anomaly_score(scores, calibration, warm, span):
    """Sensor scores [sensors, samples] to one smoothed score of the run [samples], normalized on calibration."""
    reference = scores[:, warm:calibration]
    median = np.median(reference, 1, keepdims=True)
    spread = 1.4826 * np.median(np.abs(reference - median), 1, keepdims=True) + 1e-9
    combined = ((scores - median) / spread).mean(0)
    alpha, smoothed, level = 2 / (span + 1), np.empty_like(combined), combined[0]
    for t, value in enumerate(combined):
        level = alpha * value + (1 - alpha) * level
        smoothed[t] = level
    return smoothed


def roc_auc(score, labels):
    """The probability that an anomalous sample scores above a normal one, ties counted half."""
    positive, negative = labels.sum(), (~labels).sum()
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    start = 0
    while start < len(score):
        end = start
        while end + 1 < len(score) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2 + 1
        start = end + 1
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def segments(labels):
    """The runs of labelled samples as (start, end) pairs, end exclusive."""
    padded = np.concatenate([[False], labels, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def measure(score, labels, calibration, warm):
    threshold = float(np.quantile(score[warm:calibration], 0.99))
    scored, truth = score[calibration:], labels[calibration:]
    alarm = scored > threshold
    hits = int((alarm & truth).sum())
    precision = hits / alarm.sum() if alarm.sum() else 0.0
    recall = hits / truth.sum() if truth.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    caught, delays = 0, []
    anomalies = segments(truth)
    for start, end in anomalies:
        inside = np.flatnonzero(alarm[start:end])
        if inside.size:
            caught += 1
            delays.append(int(inside[0]))
    onsets = alarm & ~np.concatenate([[False], alarm[:-1]])
    normal = (~truth).sum()
    return {"auc": roc_auc(scored, truth), "threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "anomalies": len(anomalies), "caught": caught, "delays": delays,
            "false_alarms_per_1000_normal": float(1000 * (onsets & ~truth).sum() / normal) if normal else None}


def load_run(path):
    record = json.loads(path.with_suffix(".json").read_text())
    arrays = np.load(path.with_suffix(".npz"))
    return record, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs=2, action="append", metavar=("NAME", "PATH"), required=True,
                        help="a name and the path (without suffix) of a telemetry_pilot.py output; all on one dataset")
    parser.add_argument("--window", type=int, default=100, help="samples behind the rolling z-score")
    parser.add_argument("--order", type=int, default=8, help="order of the autoregressive forecast")
    parser.add_argument("--span", type=int, help="span of the smoothing (default: 12 for nab, 5 for skab)")
    parser.add_argument("--warm", type=int, default=100,
                        help="calibration samples left out of the threshold while a stream warms up")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--scores", type=Path, help="keep every method's smoothed score of every run in this .npz")
    args = parser.parse_args()

    loaded = [(name, *load_run(Path(path))) for name, path in args.run]
    dataset = {record["dataset"] for _, record, _ in loaded}
    if len(dataset) != 1:
        parser.error("all runs are of one dataset")
    dataset = dataset.pop()
    span = args.span or (12 if dataset == "nab" else 5)
    base_record, base_arrays = loaded[0][1], loaded[0][2]
    methods = [name for name, _, _ in loaded] + list(BASELINES)
    results = {method: [] for method in methods}
    kept = {}
    for entry in base_record["runs"]:
        key, calibration = entry["key"], base_record["calibration_samples"]
        values, labels = base_arrays[f"{key}_values"], base_arrays[f"{key}_labels"].astype(bool)
        sensor_scores = {name: arrays[f"{key}_bits"] for name, record, arrays in loaded
                         if record["calibration_samples"] == calibration}
        if len(sensor_scores) != len(loaded):
            parser.error("the runs were calibrated on different stretches")
        sensor_scores["zscore"] = rolling_zscore(values, args.window)
        sensor_scores["ewma"] = ewma_residuals(values)
        sensor_scores["ar"] = ar_residuals(values, args.order, calibration)
        for method in methods:
            score = anomaly_score(sensor_scores[method], calibration, args.warm, span)
            kept[f"{entry['name']}|{method}"] = score
            results[method].append({"run": entry["name"], **measure(score, labels, calibration, args.warm)})

    summary = {}
    for method in methods:
        rows = results[method]
        aucs = [row["auc"] for row in rows if row["auc"] is not None]
        delays = [delay for row in rows for delay in row["delays"]]
        anomalies = sum(row["anomalies"] for row in rows)
        summary[method] = {"runs": len(rows), "mean_auc": float(np.mean(aucs)) if aucs else None,
                           "mean_f1": float(np.mean([row["f1"] for row in rows])),
                           "caught": sum(row["caught"] for row in rows), "anomalies": anomalies,
                           "median_delay_samples": float(np.median(delays)) if delays else None,
                           "mean_false_alarms_per_1000_normal": float(np.mean(
                               [row["false_alarms_per_1000_normal"] for row in rows
                                if row["false_alarms_per_1000_normal"] is not None]))}
    # Every Volley run against the best baseline by mean AUC, paired over the runs of the dataset.
    best = max(BASELINES, key=lambda method: summary[method]["mean_auc"] or 0)
    paired = {}
    for name, _, _ in loaded:
        mine = np.array([row["auc"] or np.nan for row in results[name]])
        theirs = np.array([row["auc"] or np.nan for row in results[best]])
        keep = np.isfinite(mine) & np.isfinite(theirs)
        difference = mine[keep] - theirs[keep]
        entry = {"against": best, "runs": int(keep.sum()), "mean_auc_difference": float(difference.mean()),
                 "runs_ahead": int((difference > 0).sum())}
        if keep.sum() > 1:
            draws = np.random.default_rng(5261).integers(keep.sum(), size=(5000, keep.sum()))
            entry["mean_auc_difference_95"] = np.percentile(difference[draws].mean(1), (2.5, 97.5)).tolist()
        paired[name] = entry

    record = {"dataset": dataset, "span": span, "window": args.window, "order": args.order, "warm": args.warm,
              "runs": {name: path for name, path in args.run}, "summary": summary, "paired_with_best_baseline": paired,
              "per_run": results}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(record, indent=2) + "\n")
    if args.scores is not None:
        np.savez_compressed(args.scores, **kept)
    print(f"{dataset}: smoothing span {span}, {summary[methods[0]]['runs']} runs")
    print(f"{'method':<22}{'AUC':>7}{'F1':>7}{'caught':>10}{'delay':>8}{'false/1k':>10}")
    for method in methods:
        row = summary[method]
        auc = "-" if row["mean_auc"] is None else f"{row['mean_auc']:.3f}"
        delay = "-" if row["median_delay_samples"] is None else f"{row['median_delay_samples']:.0f}"
        print(f"{method:<22}{auc:>7}{row['mean_f1']:>7.3f}{row['caught']:>5}/{row['anomalies']:<4}{delay:>8}"
              f"{row['mean_false_alarms_per_1000_normal']:>10.2f}")
    for name, entry in paired.items():
        interval = entry.get("mean_auc_difference_95")
        extra = "" if interval is None else f" [{interval[0]:+.3f}, {interval[1]:+.3f}]"
        print(f"{name} - {entry['against']}: {entry['mean_auc_difference']:+.3f} mean AUC{extra}, "
              f"ahead on {entry['runs_ahead']} of {entry['runs']} runs")


if __name__ == "__main__":
    main()
