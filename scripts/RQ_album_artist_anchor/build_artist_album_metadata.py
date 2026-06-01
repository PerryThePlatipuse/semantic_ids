#!/usr/bin/env python3
import argparse
import json
import os

import polars as pl


ITEM_FILES = (
    "dvae_train_items.parquet",
    "dvae_holdout_items.parquet",
    "dvae_cold_items.parquet",
)


def _prepare_mapping(path, raw_col, cluster_col, num_classes, seed):
    mapping = pl.read_parquet(path)
    required = {"item_id", raw_col}
    if not required.issubset(mapping.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    if num_classes < 2:
        raise ValueError(f"{cluster_col} needs at least two classes, including the unknown class")

    return (
        mapping
        .select("item_id", raw_col)
        .drop_nulls()
        .group_by("item_id")
        .agg(pl.col(raw_col).min())
        .with_columns(
            (
                pl.col(raw_col).cast(pl.String).hash(seed=seed) % (num_classes - 1) + 1
            ).cast(pl.Int64).alias(cluster_col)
        )
    )


def main(args):
    artist_path = args.artist_mapping or os.path.join(args.src_dir, "artist_item_mapping.parquet")
    album_path = args.album_mapping or os.path.join(args.src_dir, "album_item_mapping.parquet")
    artists = _prepare_mapping(
        artist_path,
        "artist_id",
        "artist_cluster_id",
        args.num_artist_classes,
        args.seed,
    )
    albums = _prepare_mapping(
        album_path,
        "album_id",
        "album_cluster_id",
        args.num_album_classes,
        args.seed,
    )

    all_items = []
    split_summaries = {}
    for filename in ITEM_FILES:
        path = os.path.join(args.data_dir, filename)
        items = pl.read_parquet(path)
        enriched = (
            items
            .drop(
                "artist_id",
                "album_id",
                "artist_cluster_id",
                "album_cluster_id",
                strict=False,
            )
            .join(artists, on="item_id", how="left")
            .join(albums, on="item_id", how="left")
            .with_columns(
                pl.col("artist_cluster_id").fill_null(0),
                pl.col("album_cluster_id").fill_null(0),
            )
        )
        enriched.write_parquet(path)
        all_items.append(enriched.select("item_id", "artist_id", "album_id", "artist_cluster_id", "album_cluster_id"))
        split_summaries[filename] = {
            "num_items": enriched.height,
            "missing_artist_items": enriched["artist_id"].null_count(),
            "missing_album_items": enriched["album_id"].null_count(),
        }

    metadata = pl.concat(all_items).unique(subset=["item_id"], keep="first")
    metadata.write_parquet(os.path.join(args.data_dir, "item_metadata.parquet"))
    summary = {
        "num_artist_classes": args.num_artist_classes,
        "num_album_classes": args.num_album_classes,
        "unknown_cluster_id": 0,
        "multi_value_policy": "minimum raw id per item",
        "splits": split_summaries,
    }
    with open(os.path.join(args.data_dir, "metadata_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="./data/yambda")
    ap.add_argument("--data-dir", default="./data/RQ_album_artist_anchor/yambda")
    ap.add_argument("--artist-mapping")
    ap.add_argument("--album-mapping")
    ap.add_argument("--num-artist-classes", type=int, default=512)
    ap.add_argument("--num-album-classes", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
