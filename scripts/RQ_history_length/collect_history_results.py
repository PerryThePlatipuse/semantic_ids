#!/usr/bin/env python3
import argparse
import csv
import json
import os


DEFAULT_RUNS = (
    "yambda:fixed:./results/RQ_history_length/yambda_fixed/seqrec_summary.json",
    "yambda:varlen:./results/RQ_history_length/yambda_varlen/seqrec_summary.json",
    "amazon:fixed:./results/RQ_history_length/amazon_fixed/seqrec_summary.json",
    "amazon:varlen:./results/RQ_history_length/amazon_varlen/seqrec_summary.json",
)
DEFAULT_METRICS = (
    "recall@10",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "ndcg@50",
    "ndcg@100",
    "coverage@100",
)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_run(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "run specs must have format '<dataset>:<method>:<seqrec_summary_path>', "
            f"got {spec!r}"
        )
    return tuple(part.strip() for part in parts)


def main(args):
    runs = [_parse_run(run.strip()) for run in args.runs.split(",") if run.strip()]
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]

    rows = []
    by_dataset_method_bin = {}
    for dataset, method, path in runs:
        if not os.path.exists(path):
            rows.append({
                "dataset": dataset,
                "method": method,
                "history_bin": None,
                "status": "missing_seqrec_summary",
                "summary_path": path,
            })
            continue

        summary = _load_json(path)
        by_history = summary.get("metrics", {}).get("by_history_length")
        if not by_history:
            rows.append({
                "dataset": dataset,
                "method": method,
                "history_bin": None,
                "status": "missing_by_history_length",
                "summary_path": path,
            })
            continue

        for history_bin in ("short", "medium", "long"):
            values = by_history.get(history_bin, {})
            row = {
                "dataset": dataset,
                "method": method,
                "history_bin": history_bin,
                "status": "ok",
                "summary_path": path,
                "n_users": values.get("n_users"),
                "min_history_events": values.get("min_history_events"),
                "mean_history_events": values.get("mean_history_events"),
                "max_history_events": values.get("max_history_events"),
            }
            row.update({metric: values.get(metric) for metric in metrics})
            rows.append(row)
            by_dataset_method_bin[(dataset, method, history_bin)] = row

    base = args.baseline
    for row in rows:
        dataset = row.get("dataset")
        history_bin = row.get("history_bin")
        if row.get("status") != "ok" or not history_bin:
            continue
        base_row = by_dataset_method_bin.get((dataset, base, history_bin))
        if not base_row:
            continue
        for metric in metrics:
            value = row.get(metric)
            base_value = base_row.get(metric)
            if value is None or base_value is None:
                row[f"delta_vs_{base}_{metric}"] = None
            else:
                row[f"delta_vs_{base}_{metric}"] = value - base_value

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "history_length_comparison.json")
    csv_path = os.path.join(args.output_dir, "history_length_comparison.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {json_path} and {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="./results/RQ_history_length")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS))
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    ap.add_argument("--baseline", default="fixed")
    main(ap.parse_args())
