#!/usr/bin/env python3
import argparse
import json
import math
import os

import polars as pl

from scripts.data.utils import preprocess_data
from scripts.data.yambda import TEST_INTERVAL


def _select_users(interactions, user_col, num_users, selection, seed):
    user_counts = interactions.group_by(user_col).len()
    if num_users is None or num_users <= 0:
        return user_counts.select(user_col)
    num_users = min(num_users, user_counts.height)
    if selection == "most_active":
        return user_counts.sort(["len", user_col], descending=[True, False]).head(num_users).select(user_col)
    return user_counts.select(user_col).sample(n=num_users, shuffle=True, seed=seed)


def _cap_interactions(interactions, user_col, max_interactions):
    if max_interactions is None or interactions.height <= max_interactions:
        return interactions
    per_user_cap = max(1, math.ceil(max_interactions / interactions[user_col].n_unique()))
    capped = (
        interactions
        .sort([user_col, "timestamp"])
        .group_by(user_col, maintain_order=True)
        .tail(per_user_cap)
    )
    if capped.height > max_interactions:
        capped = capped.sort("timestamp", descending=True).head(max_interactions)
    return capped.sort([user_col, "timestamp"])


def main(args):
    os.makedirs(args.dst_dir, exist_ok=True)
    interactions = pl.read_parquet(os.path.join(args.src_dir, "interactions.parquet"))
    embeddings = pl.read_parquet(os.path.join(args.src_dir, "embeddings.parquet"))

    user_col = "uid" if "uid" in interactions.columns else "user_id"
    required = {user_col, "item_id", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(f"interactions must contain columns: {sorted(required)}")
    if not {"item_id", "embed"}.issubset(embeddings.columns):
        raise ValueError("embeddings must contain columns: ['embed', 'item_id']")

    interactions = interactions.join(embeddings.select("item_id"), on="item_id", how="semi")
    users = _select_users(interactions, user_col, args.num_users, args.user_selection, args.seed)
    subset = interactions.join(users, on=user_col, how="semi")
    subset = _cap_interactions(subset, user_col, args.max_interactions)
    if user_col != "user_id":
        subset = subset.rename({user_col: "user_id"})

    max_timestamp = subset["timestamp"].max()
    train = subset.filter(pl.col("timestamp") < max_timestamp - args.test_interval)
    test = subset.filter(pl.col("timestamp") >= max_timestamp - args.test_interval)
    if train.height == 0 or test.height == 0:
        raise ValueError("temporal split produced an empty train or test split")

    subset_items = subset.select("item_id").unique()
    subset_embeddings = embeddings.join(subset_items, on="item_id", how="semi")
    subset.write_parquet(os.path.join(args.dst_dir, "subset_interactions.parquet"))
    subset_embeddings.write_parquet(os.path.join(args.dst_dir, "embeddings.parquet"))

    preprocess_data(
        train=train,
        test=test,
        item_embeddings=subset_embeddings,
        dst_dir=args.dst_dir,
        core_threshold=args.core_threshold,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        verbose=True,
        topk_head=args.topk_head,
        max_core_items=args.max_core_items,
    )

    seqrec_test_path = os.path.join(args.dst_dir, "seqrec_test_interactions.parquet")
    seqrec_test = pl.read_parquet(seqrec_test_path)
    sampled_users = (
        seqrec_test.select("user_id")
        .unique()
        .sample(fraction=args.seqrec_test_user_fraction, shuffle=True, seed=args.seed)
    )
    seqrec_test.join(sampled_users, on="user_id", how="semi").write_parquet(
        os.path.join(args.dst_dir, "seqrec_test_sample_interactions.parquet")
    )

    summary = {
        "src_dir": args.src_dir,
        "dst_dir": args.dst_dir,
        "user_selection": args.user_selection,
        "num_users": subset["user_id"].n_unique(),
        "num_interactions": subset.height,
        "num_items": subset["item_id"].n_unique(),
        "num_train_interactions": train.height,
        "num_test_interactions": test.height,
        "max_core_items": args.max_core_items,
        "seed": args.seed,
    }
    with open(os.path.join(args.dst_dir, "subset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="./data/yambda")
    ap.add_argument("--dst-dir", default="./data/RQ_album_artist_anchor/yambda")
    # By default, keep the full Yambda scale used by the paper configs.
    # Pass positive values to build a cheaper local subset.
    ap.add_argument("--num-users", type=int, default=None)
    ap.add_argument("--max-interactions", type=int, default=None)
    ap.add_argument("--max-core-items", type=int, default=None)
    ap.add_argument("--core-threshold", type=int, default=16)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--topk-head", type=int, default=30000)
    ap.add_argument("--seqrec-test-user-fraction", type=float, default=0.05)
    ap.add_argument("--user-selection", choices=("most_active", "random"), default="most_active")
    ap.add_argument("--test-interval", type=int, default=TEST_INTERVAL)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
