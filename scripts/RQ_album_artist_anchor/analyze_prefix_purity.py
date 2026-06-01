#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict

import polars as pl


def _percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    return float(values[max(0, math.ceil(q * len(values)) - 1)])


def _prefix_purity(records, label_col, max_depth):
    result = {}
    for depth in range(1, max_depth + 1):
        buckets = defaultdict(Counter)
        for row in records:
            if len(row["sid"]) >= depth:
                buckets[row["sid"][:depth]][row[label_col]] += 1
        num_items = sum(sum(labels.values()) for labels in buckets.values())
        majority_items = sum(max(labels.values()) for labels in buckets.values())
        result[str(depth)] = {
            "num_items": num_items,
            "num_prefixes": len(buckets),
            "purity": majority_items / num_items if num_items else None,
        }
    return result


def main(args):
    sids = pl.read_parquet(args.sids)
    metadata = pl.read_parquet(args.metadata)
    required_sids = {"item_id", "sid", "length"}
    required_metadata = {"item_id", args.artist_col, args.album_col}
    if not required_sids.issubset(sids.columns):
        raise ValueError(f"SIDs must contain columns: {sorted(required_sids)}")
    if not required_metadata.issubset(metadata.columns):
        raise ValueError(f"metadata must contain columns: {sorted(required_metadata)}")

    joined = sids.select("item_id", "sid", "length").join(
        metadata.select("item_id", args.artist_col, args.album_col),
        on="item_id",
        how="inner",
    )
    records = []
    for row in joined.iter_rows(named=True):
        length = int(row["length"])
        records.append({
            "sid": tuple(int(token) for token in row["sid"][:length]),
            args.artist_col: int(row[args.artist_col]),
            args.album_col: int(row[args.album_col]),
        })

    sid_buckets = Counter(row["sid"] for row in records)
    bucket_sizes = list(sid_buckets.values())
    lengths = [len(row["sid"]) for row in records]
    max_depth = max(lengths, default=0)
    output = {
        "num_items": len(records),
        "num_unique_sids": len(sid_buckets),
        "mean_sid_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "collision_bucket_count": sum(size > 1 for size in bucket_sizes),
        "colliding_item_count": sum(size for size in bucket_sizes if size > 1),
        "excess_item_collisions": sum(size - 1 for size in bucket_sizes),
        "mean_items_per_sid": sum(bucket_sizes) / len(bucket_sizes) if bucket_sizes else 0.0,
        "p95_items_per_sid": _percentile(bucket_sizes, 0.95),
        "max_items_per_sid": max(bucket_sizes, default=0),
        "artist_prefix_purity": _prefix_purity(records, args.artist_col, max_depth),
        "album_prefix_purity": _prefix_purity(records, args.album_col, max_depth),
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", required=True)
    ap.add_argument("--metadata", default="./data/RQ_album_artist_anchor/yambda/item_metadata.parquet")
    ap.add_argument("--output", required=True)
    ap.add_argument("--artist-col", default="artist_cluster_id")
    ap.add_argument("--album-col", default="album_cluster_id")
    main(ap.parse_args())
