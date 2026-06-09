#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import polars as pl


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_") or "encoder"


def _parse_embedding_specs(values: Iterable[str], embeddings_dir: str | None) -> List[Tuple[str, str]]:
    specs = []
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
    if not specs:
        raise ValueError("pass --embedding name=path or --embeddings-dir")
    return specs


def _pick_col(df: pl.DataFrame, requested: str | None, candidates: Tuple[str, ...], kind: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"{kind} column {requested!r} is missing; columns={df.columns}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"cannot infer {kind} column; tried {candidates}, columns={df.columns}")


def _sample_matrix(path: str, embed_col: str, sample_size: int) -> Tuple[np.ndarray, int]:
    sample = pl.read_parquet(path, columns=[embed_col], n_rows=sample_size)
    sample_rows = sample.height
    sample = sample.filter(pl.col(embed_col).is_not_null())
    return np.asarray(sample[embed_col].to_list(), dtype=np.float32), sample_rows


def _safe_quantiles(values: np.ndarray, qs: List[float]) -> List[float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [math.nan for _ in qs]
    return [float(x) for x in np.quantile(values, qs)]


def check_one(name: str, path: str, args: argparse.Namespace) -> dict:
    head = pl.read_parquet(path, n_rows=1)
    item_col = _pick_col(head, args.item_col, ("item_id", "parent_asin", "asin"), "item")
    embed_col = _pick_col(head, args.embed_col, ("embed", "embedding", "embeddings"), "embedding")

    rows = int(pl.scan_parquet(path).select(pl.len()).collect().item())
    items = pl.read_parquet(path, columns=[item_col])
    item_unique = items[item_col].n_unique()
    null_items = int(items.select(pl.col(item_col).is_null().sum()).item())

    first = pl.read_parquet(path, columns=[embed_col], n_rows=1)
    if first.height == 0:
        raise ValueError(f"{path} has no non-null embeddings")
    dim = len(first[embed_col][0])

    X, sample_rows = _sample_matrix(path, embed_col, args.sample_size)
    norms = np.linalg.norm(X, axis=1)
    finite_rows = np.isfinite(X).all(axis=1)

    Y = X[finite_rows]
    cos_quantiles = [math.nan, math.nan, math.nan]
    if len(Y) >= 2:
        Y = Y / np.clip(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12, None)
        rng = np.random.default_rng(args.seed)
        pair_n = min(args.cos_pairs, len(Y) * (len(Y) - 1) // 2)
        left = rng.integers(0, len(Y), size=pair_n)
        right = rng.integers(0, len(Y), size=pair_n)
        mask = left != right
        cos = (Y[left[mask]] * Y[right[mask]]).sum(axis=1)
        cos_quantiles = _safe_quantiles(cos, [0.01, 0.5, 0.99])

    result = {
        "encoder": name,
        "path": path,
        "rows": rows,
        "item_col": item_col,
        "embed_col": embed_col,
        "unique_items": int(item_unique),
        "duplicate_item_rows": int(rows - item_unique),
        "null_items": null_items,
        "null_embeddings_in_sample": int(sample_rows - len(X)),
        "embed_dim": int(dim),
        "sample_size": int(len(X)),
        "finite_sample_rows": int(finite_rows.sum()),
        "norm_min": float(np.nanmin(norms)),
        "norm_mean": float(np.nanmean(norms)),
        "norm_max": float(np.nanmax(norms)),
        "mean_dim_std": float(np.nanmean(X.std(axis=0))),
        "cos_q01": cos_quantiles[0],
        "cos_q50": cos_quantiles[1],
        "cos_q99": cos_quantiles[2],
    }
    return result


def main(args: argparse.Namespace) -> None:
    rows = []
    for name, path in _parse_embedding_specs(args.embedding, args.embeddings_dir):
        result = check_one(name, path, args)
        rows.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Check embedding parquet files")
    ap.add_argument("--embedding", action="append", default=[], help="Encoder spec: name=path.parquet. May be repeated.")
    ap.add_argument("--embeddings-dir", default=None)
    ap.add_argument("--item-col", default=None)
    ap.add_argument("--embed-col", default=None)
    ap.add_argument("--sample-size", type=int, default=10000)
    ap.add_argument("--cos-pairs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    main(ap.parse_args())
