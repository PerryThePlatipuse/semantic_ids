#!/usr/bin/env python3
import argparse
import csv
import json
import os


METHODS = ("original", "aux", "prefix")
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
    "head_recall@10",
    "tail_recall@10",
    "long_tail_share",
)


def _load_optional(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _purity_at_one(purity, key):
    if not purity:
        return None
    return purity.get(key, {}).get("1", {}).get("purity")


def main(args):
    rows = []
    for method in METHODS:
        method_dir = os.path.join(args.results_dir, method)
        dvae = _load_optional(os.path.join(method_dir, "dvae_metrics.json"))
        seqrec = _load_optional(os.path.join(method_dir, "seqrec_summary.json"))
        purity = _load_optional(os.path.join(method_dir, "prefix_purity.json"))
        seqrec_metrics = seqrec.get("metrics", {}) if seqrec else {}
        row = {
            "method": method,
            "mean_sid_length": purity.get("mean_sid_length") if purity else None,
            "num_unique_sids": purity.get("num_unique_sids") if purity else None,
            "excess_item_collisions": purity.get("excess_item_collisions") if purity else None,
            "mean_items_per_sid": purity.get("mean_items_per_sid") if purity else None,
            "artist_prefix_purity@1": _purity_at_one(purity, "artist_prefix_purity"),
            "album_prefix_purity@1": _purity_at_one(purity, "album_prefix_purity"),
            "has_dvae_metrics": dvae is not None,
            "has_seqrec_summary": seqrec is not None,
        }
        row.update({name: seqrec_metrics.get(name) for name in SEQREC_METRICS})
        rows.append(row)

    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(args.results_dir, "comparison.json")
    csv_path = os.path.join(args.results_dir, "comparison.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {json_path} and {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results/RQ_album_artist_anchor")
    main(ap.parse_args())
