#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import polars as pl
import yaml

from scripts.data.utils import preprocess_data


DEFAULT_DVAE_CONFIG = "configs/RQ_encoder_comparison/amazon_toys_fixed_dvae.yaml"
DEFAULT_SEQREC_CONFIG = "configs/RQ_encoder_comparison/seqrec_amazon_toys_fixed.yaml"
SECONDS_PER_DAY = 24 * 60 * 60


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("empty encoder name after slugify")
    return value


def _load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, cfg: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run(args: List[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def _parse_embedding_specs(values: Iterable[str], embeddings_dir: str | None) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        specs.append((_slugify(name), path))

    if embeddings_dir:
        for path in sorted(Path(embeddings_dir).glob("*.parquet")):
            specs.append((_slugify(path.stem), str(path)))

    seen = set()
    unique_specs = []
    for name, path in specs:
        if name in seen:
            raise ValueError(f"duplicate encoder name: {name}")
        seen.add(name)
        unique_specs.append((name, path))
    if not unique_specs:
        raise ValueError("pass at least one --embedding name=path or --embeddings-dir")
    return unique_specs


def _pick_col(df: pl.DataFrame, requested: str | None, candidates: Tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{kind} column {requested!r} is missing; columns={df.columns}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"cannot infer {kind} column; tried {candidates}, columns={df.columns}")


def _normalize_timestamp(df: pl.DataFrame, unit: str) -> pl.DataFrame:
    if unit == "s":
        return df
    if unit == "ms":
        return df.with_columns((pl.col("timestamp") // 1000).alias("timestamp"))
    if unit != "auto":
        raise ValueError("timestamp_unit must be one of: auto, s, ms")

    max_ts = int(df["timestamp"].max())
    if max_ts > 10_000_000_000:
        return df.with_columns((pl.col("timestamp") // 1000).alias("timestamp"))
    return df


def _load_interactions(args: argparse.Namespace) -> pl.DataFrame:
    df = pl.read_parquet(args.interactions)
    user_col = _pick_col(df, args.user_col, ("user_id", "reviewerID", "uid"), "user")
    item_col = _pick_col(df, args.item_col, ("item_id", "parent_asin", "asin"), "item")
    time_col = _pick_col(df, args.time_col, ("timestamp", "unixReviewTime", "time"), "time")

    if args.rating_col and args.rating_col in df.columns and args.min_rating is not None:
        df = df.filter(pl.col(args.rating_col) >= float(args.min_rating))

    df = (
        df.select([
            pl.col(user_col).cast(pl.String).alias("user_id"),
            pl.col(item_col).cast(pl.String).alias("item_id"),
            pl.col(time_col).cast(pl.Int64).alias("timestamp"),
        ])
        .drop_nulls(["user_id", "item_id", "timestamp"])
    )
    df = _normalize_timestamp(df, args.timestamp_unit)

    if args.max_users is not None:
        users = (
            df.select("user_id")
            .unique()
            .sample(n=min(args.max_users, df["user_id"].n_unique()), shuffle=True, seed=args.seed)
        )
        df = df.join(users, on="user_id", how="semi")

    if args.max_interactions is not None and df.height > args.max_interactions:
        df = df.sample(n=args.max_interactions, shuffle=True, seed=args.seed)

    return df


def _load_embeddings(path: str, item_col: str | None, embed_col: str | None) -> Tuple[pl.DataFrame, int]:
    df = pl.read_parquet(path)
    item_col = _pick_col(df, item_col, ("item_id", "parent_asin", "asin"), "embedding item")
    embed_col = _pick_col(df, embed_col, ("embed", "embedding", "embeddings"), "embedding")
    out = (
        df.select([
            pl.col(item_col).cast(pl.String).alias("item_id"),
            pl.col(embed_col).alias("embed"),
        ])
        .drop_nulls(["item_id", "embed"])
        .unique(subset=["item_id"], keep="first")
    )
    if out.height == 0:
        raise ValueError(f"no embeddings loaded from {path}")
    dim = len(out["embed"][0])
    return out, int(dim)


def _build_test_sample(data_dir: Path, user_fraction: float, seed: int) -> None:
    src = data_dir / "seqrec_test_interactions.parquet"
    dst = data_dir / "seqrec_test_sample_interactions.parquet"
    test = pl.read_parquet(src)
    if test.height == 0:
        test.write_parquet(dst)
        print(f"Saved {dst}: 0 rows, 0 users")
        return
    n_users = test["user_id"].n_unique()
    sample_n = max(1, int(np.ceil(n_users * user_fraction)))
    users = (
        test.select("user_id")
        .unique()
        .sample(n=min(sample_n, n_users), shuffle=True, seed=seed)
    )
    sampled = test.join(users, on="user_id", how="semi")
    sampled.write_parquet(dst)
    print(f"Saved {dst}: {sampled.height:,} rows, {sampled['user_id'].n_unique():,} users")


def _prepare_data(
    interactions: pl.DataFrame,
    embeddings: pl.DataFrame,
    data_dir: Path,
    args: argparse.Namespace,
) -> None:
    max_ts = int(interactions["timestamp"].max())
    test_start = max_ts - int(args.test_interval_days * SECONDS_PER_DAY)
    train = interactions.filter(pl.col("timestamp") < test_start)
    test = interactions.filter(pl.col("timestamp") >= test_start)
    if train.height == 0 or test.height == 0:
        raise ValueError(
            f"empty train/test split: train={train.height}, test={test.height}, "
            f"test_start={test_start}"
        )

    preprocess_data(
        train=train,
        test=test,
        item_embeddings=embeddings,
        dst_dir=str(data_dir),
        core_threshold=args.core_threshold,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        verbose=True,
        topk_head=args.pop_k,
        max_core_items=args.max_core_items,
    )
    _build_test_sample(data_dir, args.test_user_fraction, args.seed)


def _sid_key(values: List[int]) -> str:
    return ",".join(str(int(x)) for x in values)


def _write_sid_metrics(sids_path: Path, output_path: Path) -> None:
    sids = pl.read_parquet(sids_path)
    sid_lens = sids["length"].to_list() if "length" in sids.columns else [len(x) for x in sids["sid"].to_list()]
    keyed = sids.with_columns(
        pl.col("sid").map_elements(_sid_key, return_dtype=pl.String).alias("sid_key")
    )
    bucket_sizes = keyed.group_by("sid_key").len().rename({"len": "bucket_size"})
    sizes = bucket_sizes["bucket_size"].to_list()
    metrics = {
        "num_items": int(sids.height),
        "num_unique_sids": int(bucket_sizes.height),
        "mean_sid_length": float(sum(sid_lens) / len(sid_lens)) if sid_lens else 0.0,
        "collision_bucket_count": int(sum(size > 1 for size in sizes)),
        "colliding_item_count": int(sum(size for size in sizes if size > 1)),
        "excess_item_collisions": int(sum(size - 1 for size in sizes if size > 1)),
        "mean_items_per_sid": float(sum(sizes) / len(sizes)) if sizes else 0.0,
        "max_items_per_sid": int(max(sizes)) if sizes else 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Saved {output_path}")


def _run_dvae(name: str, data_dir: Path, out_dir: Path, embed_dim: int, args: argparse.Namespace) -> None:
    cfg = _load_yaml(args.dvae_config)
    cfg["data_dir"] = str(data_dir)
    cfg["out_dir"] = str(out_dir)
    cfg["metrics_filename"] = "dvae_metrics.json"
    cfg["embed_dim"] = int(embed_dim)
    cfg["seed"] = int(args.seed)
    if args.num_epochs is not None:
        cfg["num_epochs"] = int(args.num_epochs)

    config_path = out_dir / "dvae_config.generated.yaml"
    _write_yaml(config_path, cfg)
    _run([sys.executable, "-m", "scripts.train_dvae", "--config", str(config_path)])


def _run_seqrec(name: str, data_dir: Path, out_dir: Path, args: argparse.Namespace) -> None:
    cfg = _load_yaml(args.seqrec_config)
    cfg["data_dir"] = str(data_dir)
    cfg["semantic_ids_path"] = str(out_dir / "sids.parquet")
    cfg["summary_json"] = str(out_dir / "seqrec_summary.json")
    cfg["seed"] = int(args.seed)

    config_path = out_dir / "seqrec_config.generated.yaml"
    _write_yaml(config_path, cfg)
    _run([sys.executable, "-m", "scripts.train_seqrec", "--config", str(config_path)])


def main(args: argparse.Namespace) -> None:
    stages = {stage.strip() for stage in args.stages.split(",") if stage.strip()}
    unknown = stages - {"prepare", "dvae", "sid_metrics", "seqrec"}
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    specs = _parse_embedding_specs(args.embedding, args.embeddings_dir)
    interactions = _load_interactions(args)
    print(f"Interactions: {interactions.height:,} rows, {interactions['user_id'].n_unique():,} users")

    for name, embedding_path in specs:
        data_dir = Path(args.work_data_dir) / name
        out_dir = Path(args.results_dir) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        embeddings, embed_dim = _load_embeddings(embedding_path, args.embedding_item_col, args.embedding_col)
        print(f"\n== {name} ==")
        print(f"Embeddings: {embeddings.height:,} items, dim={embed_dim}, path={embedding_path}")

        if "prepare" in stages:
            _prepare_data(interactions, embeddings, data_dir, args)
        if "dvae" in stages:
            _run_dvae(name, data_dir, out_dir, embed_dim, args)
        if "sid_metrics" in stages:
            _write_sid_metrics(out_dir / "sids.parquet", out_dir / "sid_metrics.json")
        if "seqrec" in stages:
            _run_seqrec(name, data_dir, out_dir, args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run fixed-dVAE + seqrec encoder comparison")
    ap.add_argument("--interactions", default="./data/amazon_toys_encoders/interactions.parquet")
    ap.add_argument("--embedding", action="append", default=[], help="Encoder spec: name=path.parquet. May be repeated.")
    ap.add_argument("--embeddings-dir", default=None, help="Directory with *.parquet embedding files.")

    ap.add_argument("--user-col", default=None)
    ap.add_argument("--item-col", default=None)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--rating-col", default="rating")
    ap.add_argument("--min-rating", type=float, default=4.0)
    ap.add_argument("--timestamp-unit", default="auto", choices=["auto", "s", "ms"])
    ap.add_argument("--embedding-item-col", default=None)
    ap.add_argument("--embedding-col", default=None)

    ap.add_argument("--work-data-dir", default="./data/RQ_encoder_comparison/amazon_toys")
    ap.add_argument("--results-dir", default="./results/RQ_encoder_comparison_amazon_toys")
    ap.add_argument("--dvae-config", default=DEFAULT_DVAE_CONFIG)
    ap.add_argument("--seqrec-config", default=DEFAULT_SEQREC_CONFIG)
    ap.add_argument("--stages", default="prepare,dvae,sid_metrics,seqrec")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-interval-days", type=int, default=28)
    ap.add_argument("--core-threshold", type=int, default=16)
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--test-user-fraction", type=float, default=0.1)
    ap.add_argument("--max-users", type=int, default=5000)
    ap.add_argument("--max-interactions", type=int, default=3_000_000)
    ap.add_argument("--max-core-items", type=int, default=30000)
    ap.add_argument("--pop-k", type=int, default=30000)
    ap.add_argument("--num-epochs", type=int, default=None)
    main(ap.parse_args())
