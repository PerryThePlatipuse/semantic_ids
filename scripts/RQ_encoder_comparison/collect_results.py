#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path


SEQREC_METRICS = (
    "recall@10",
    "recall@50",
    "recall@100",
    "ndcg@10",
    "ndcg@50",
    "ndcg@100",
    "coverage@10",
    "coverage@50",
    "coverage@100",
    "tail_recall@10",
    "tail_recall@50",
    "tail_recall@100",
    "long_tail_share",
)


def _load_optional(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(data, path):
    cur = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main(args):
    rows = []
    results_dir = Path(args.results_dir)
    encoders = [x.strip() for x in args.encoders.split(",") if x.strip()]
    if not encoders:
        encoders = sorted(path.name for path in results_dir.iterdir() if path.is_dir())

    for encoder in encoders:
        out_dir = results_dir / encoder
        dvae = _load_optional(out_dir / "dvae_metrics.json") or {}
        sid = _load_optional(out_dir / "sid_metrics.json") or {}
        seqrec = _load_optional(out_dir / "seqrec_summary.json") or {}
        seqrec_metrics = seqrec.get("metrics", {})

        row = {
            "encoder": encoder,
            "has_dvae_metrics": bool(dvae),
            "has_sid_metrics": bool(sid),
            "has_seqrec_summary": bool(seqrec),
            "dvae_train_ndcg": _get_nested(dvae, ("train_holdout", "ndcg")),
            "dvae_cold_ndcg": _get_nested(dvae, ("cold", "ndcg")),
            "mean_sid_length": sid.get("mean_sid_length"),
            "num_unique_sids": sid.get("num_unique_sids"),
            "collision_bucket_count": sid.get("collision_bucket_count"),
            "excess_item_collisions": sid.get("excess_item_collisions"),
            "mean_items_per_sid": sid.get("mean_items_per_sid"),
            "max_items_per_sid": sid.get("max_items_per_sid"),
        }
        row.update({metric: seqrec_metrics.get(metric) for metric in SEQREC_METRICS})
        rows.append(row)

    os.makedirs(results_dir, exist_ok=True)
    json_path = results_dir / "comparison.json"
    csv_path = results_dir / "comparison.csv"
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
    ap.add_argument("--results-dir", default="./results/RQ_encoder_comparison")
    ap.add_argument("--encoders", default="", help="Comma-separated encoder names. Empty means all result dirs.")
    main(ap.parse_args())
