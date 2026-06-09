#!/usr/bin/env python3
import argparse
import gzip
import io
import json
import os
import re
import sys
import types
from importlib.machinery import ModuleSpec
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_ENCODERS = (
    "tiny_minilm_l3=sentence-transformers/paraphrase-MiniLM-L3-v2",
    "minilm_l6=sentence-transformers/all-MiniLM-L6-v2",
    "bge_small=BAAI/bge-small-en-v1.5",
)


def _install_transformers_optional_dependency_stubs() -> None:
    """Avoid broken optional deps that transformers imports but encoder inference does not use."""

    def unavailable(*args, **kwargs):
        raise RuntimeError("Optional dependency stub called during text embedding inference")

    if "sklearn.metrics" not in sys.modules:
        sklearn = types.ModuleType("sklearn")
        sklearn.__spec__ = ModuleSpec("sklearn", loader=None, is_package=True)
        sklearn.__path__ = []

        metrics = types.ModuleType("sklearn.metrics")
        metrics.__spec__ = ModuleSpec("sklearn.metrics", loader=None)
        metrics.roc_curve = unavailable

        sklearn.metrics = metrics
        sys.modules.setdefault("sklearn", sklearn)
        sys.modules.setdefault("sklearn.metrics", metrics)

    for name in ("soxr", "librosa", "soundfile"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__spec__ = ModuleSpec(name, loader=None)
            module.resample = unavailable
            module.load = unavailable
            module.read = unavailable
            sys.modules[name] = module


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    if not value:
        raise ValueError("empty encoder name")
    return value


def _parse_encoder_specs(values: Iterable[str]) -> List[Tuple[str, str]]:
    specs = []
    for value in values:
        if "=" in value:
            name, model_id = value.split("=", 1)
        else:
            model_id = value
            name = model_id.rsplit("/", 1)[-1]
        specs.append((_slugify(name), model_id))
    return specs


def _read_gzip_to_ram(path: str) -> bytes:
    path = str(path)
    compressed = Path(path).read_bytes()
    print(f"Loaded compressed {path}: {len(compressed) / 1e6:.1f} MB")
    data = gzip.decompress(compressed)
    print(f"Decompressed in RAM: {len(data) / 1e6:.1f} MB")
    return data


def _iter_jsonl_bytes(data: bytes, desc: str):
    with io.BytesIO(data) as fh:
        for line in tqdm(fh, desc=desc, dynamic_ncols=True):
            if line.strip():
                yield json.loads(line)


def _as_text(value, max_parts: int = 8) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_as_text(x, max_parts=max_parts) for x in value[:max_parts]]
        return " | ".join(x for x in parts if x)
    if isinstance(value, dict):
        parts = []
        for i, (key, val) in enumerate(value.items()):
            if i >= max_parts:
                break
            text = _as_text(val, max_parts=max_parts)
            if text:
                parts.append(f"{key}: {text}")
        return " | ".join(parts)
    return str(value)


def _build_item_text(row: Dict) -> str:
    pieces = []
    fields = [
        ("Title", row.get("title")),
        ("Main category", row.get("main_category")),
        ("Store", row.get("store")),
        ("Categories", row.get("categories")),
        ("Features", row.get("features")),
        ("Description", row.get("description")),
        ("Details", row.get("details")),
    ]
    for label, value in fields:
        text = _as_text(value)
        if text:
            pieces.append(f"{label}: {text}")
    if not pieces:
        pieces.append(f"Item: {row.get('parent_asin') or row.get('asin') or ''}")
    return " ; ".join(pieces)


def _load_reviews(args: argparse.Namespace) -> pl.DataFrame:
    data = _read_gzip_to_ram(args.reviews_gz)
    rows = []
    for row in _iter_jsonl_bytes(data, "parse reviews"):
        rating = row.get(args.rating_col)
        if rating is None or float(rating) < args.min_rating:
            continue
        user_id = row.get(args.user_col)
        item_id = row.get(args.item_col)
        timestamp = row.get(args.time_col)
        if user_id is None or item_id is None or timestamp is None:
            continue
        rows.append({
            "user_id": str(user_id),
            "item_id": str(item_id),
            "timestamp": int(timestamp),
            "rating": float(rating),
        })
    del data

    if not rows:
        raise ValueError("no review rows after filtering")

    df = pl.DataFrame(rows)
    if args.timestamp_unit == "ms":
        df = df.with_columns((pl.col("timestamp") // 1000).alias("timestamp"))
    elif args.timestamp_unit == "auto" and int(df["timestamp"].max()) > 10_000_000_000:
        df = df.with_columns((pl.col("timestamp") // 1000).alias("timestamp"))
    elif args.timestamp_unit != "s" and args.timestamp_unit != "auto":
        raise ValueError("timestamp_unit must be one of: auto, s, ms")

    if args.user_selection == "most_active":
        users = (
            df.group_by("user_id")
            .len()
            .sort(["len", "user_id"], descending=[True, False])
            .head(args.max_users)
            .select("user_id")
        )
    else:
        users = (
            df.select("user_id")
            .unique()
            .sample(n=min(args.max_users, df["user_id"].n_unique()), shuffle=True, seed=args.seed)
        )
    df = df.join(users, on="user_id", how="semi")

    if args.max_interactions is not None and df.height > args.max_interactions:
        per_user_cap = max(1, int(np.ceil(args.max_interactions / df["user_id"].n_unique())))
        df = (
            df.sort(["user_id", "timestamp"])
            .group_by("user_id", maintain_order=True)
            .tail(per_user_cap)
        )
        if df.height > args.max_interactions:
            df = df.sort("timestamp", descending=True).head(args.max_interactions)

    df = df.sort(["user_id", "timestamp", "item_id"])
    print(
        f"Reviews after filtering: {df.height:,} rows, "
        f"{df['user_id'].n_unique():,} users, {df['item_id'].n_unique():,} items"
    )
    return df


def _load_item_texts(args: argparse.Namespace, item_ids: set) -> pl.DataFrame:
    data = _read_gzip_to_ram(args.meta_gz)
    seen = set()
    rows = []
    for row in _iter_jsonl_bytes(data, "parse metadata"):
        item_id = row.get("parent_asin") or row.get("asin")
        if item_id is None:
            continue
        item_id = str(item_id)
        if item_id not in item_ids or item_id in seen:
            continue
        seen.add(item_id)
        rows.append({
            "item_id": item_id,
            "text": _build_item_text(row),
            "title": _as_text(row.get("title")),
            "main_category": _as_text(row.get("main_category")),
            "store": _as_text(row.get("store")),
        })
    del data

    if not rows:
        raise ValueError("metadata produced no item texts")

    df = pl.DataFrame(rows).sort("item_id")
    missing = len(item_ids) - df.height
    print(f"Item texts: {df.height:,}; missing metadata for {missing:,} interaction items")
    return df


def _login_hf(args: argparse.Namespace) -> None:
    token = args.hf_token or os.environ.get(args.hf_token_env)
    if not token:
        return
    from huggingface_hub import login

    login(token=token)


def _encode_one(
    name: str,
    model_id: str,
    texts: List[str],
    item_ids: List[str],
    args: argparse.Namespace,
) -> pl.DataFrame:
    _install_transformers_optional_dependency_stubs()
    from transformers import AutoModel, AutoTokenizer

    print(f"\nEncoding {name}: {model_id}")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=args.trust_remote_code)
    model = model.to(device)
    model.eval()

    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc=f"encode {name}", dynamic_ncols=True):
            batch_texts = texts[start:start + args.batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device, non_blocking=True) for key, value in encoded.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                output = model(**encoded)
                token_embeddings = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
                emb = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                emb = F.normalize(emb.float(), dim=-1)
            chunks.append(emb.cpu().numpy().astype(np.float32, copy=False))

    embeddings = np.concatenate(chunks, axis=0)
    if embeddings.ndim != 2:
        raise ValueError(f"{name} embeddings must be 2D, got {embeddings.shape}")
    dim = int(embeddings.shape[1])
    df = pl.DataFrame({"item_id": item_ids}).with_columns(
        pl.Series("embed", embeddings.tolist(), dtype=pl.List(pl.Float32))
    ).with_columns(
        pl.col("embed").cast(pl.Array(pl.Float32, dim))
    )
    print(f"{name}: {len(item_ids):,} items, dim={dim}")
    return df


def _write_summary(args: argparse.Namespace, interactions: pl.DataFrame, item_texts: pl.DataFrame, encoders: List[Tuple[str, str]]) -> None:
    summary = {
        "reviews_gz": args.reviews_gz,
        "meta_gz": args.meta_gz,
        "output_dir": args.output_dir,
        "min_rating": args.min_rating,
        "max_users": args.max_users,
        "max_interactions": args.max_interactions,
        "user_selection": args.user_selection,
        "num_interactions": interactions.height,
        "num_users": interactions["user_id"].n_unique(),
        "num_items_in_interactions": interactions["item_id"].n_unique(),
        "num_item_texts": item_texts.height,
        "encoders": [{"name": name, "model_id": model_id} for name, model_id in encoders],
    }
    out = Path(args.output_dir) / "build_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Saved {out}")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    encoders = _parse_encoder_specs(args.encoder or DEFAULT_ENCODERS)
    _login_hf(args)

    interactions_path = output_dir / "interactions.parquet"
    item_texts_path = output_dir / "item_texts.parquet"

    if args.reuse_prepared and interactions_path.exists() and item_texts_path.exists():
        interactions = pl.read_parquet(interactions_path)
        item_texts = pl.read_parquet(item_texts_path)
        print(f"Reusing {interactions_path}")
        print(f"Reusing {item_texts_path}")
    else:
        interactions = _load_reviews(args)
        interactions.write_parquet(interactions_path)
        print(f"Saved {interactions_path}")

        item_ids = set(interactions["item_id"].to_list())
        item_texts = _load_item_texts(args, item_ids)
        item_texts.write_parquet(item_texts_path)
        print(f"Saved {item_texts_path}")

    item_ids = item_texts["item_id"].to_list()
    texts = item_texts["text"].to_list()
    for name, model_id in encoders:
        out_path = embeddings_dir / f"{name}.parquet"
        if args.reuse_embeddings and out_path.exists():
            print(f"Reusing {out_path}")
            continue
        emb_df = _encode_one(name, model_id, texts, item_ids, args)
        emb_df.write_parquet(out_path)
        print(f"Saved {out_path}")

    _write_summary(args, interactions, item_texts, encoders)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build Amazon Toys interactions/text embeddings with RAM-first gzip reads")
    ap.add_argument("--reviews-gz", default="data/Toys_and_Games.jsonl.gz")
    ap.add_argument("--meta-gz", default="data/meta_Toys_and_Games.jsonl.gz")
    ap.add_argument("--output-dir", default="data/amazon_toys_encoders")
    ap.add_argument("--encoder", action="append", default=[], help="name=model_id. May be repeated.")
    ap.add_argument("--hf-token", default=None)
    ap.add_argument("--hf-token-env", default="HF_TOKEN")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--reuse-prepared", action="store_true")
    ap.add_argument("--reuse-embeddings", action="store_true")

    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--item-col", default="parent_asin")
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--rating-col", default="rating")
    ap.add_argument("--timestamp-unit", default="auto", choices=["auto", "s", "ms"])
    ap.add_argument("--min-rating", type=float, default=4.0)
    ap.add_argument("--max-users", type=int, default=5000)
    ap.add_argument("--max-interactions", type=int, default=3_000_000)
    ap.add_argument("--user-selection", choices=["most_active", "random"], default="most_active")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
