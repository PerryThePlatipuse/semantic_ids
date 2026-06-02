#!/usr/bin/env python3
import argparse
import copy
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import yaml

from lib.utils import configure_torch


def require(cfg, key):
    if key not in cfg:
        raise KeyError(f"Missing required config key: {key}")
    return cfg[key]


@torch.no_grad()
def assign_labels(x, centroids, batch_size=8192):
    labels = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    c_norm = centroids.pow(2).sum(dim=1)[None, :]
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        xb = x[start:end]
        dists = xb.pow(2).sum(dim=1, keepdim=True) + c_norm - 2.0 * xb @ centroids.T
        labels[start:end] = dists.argmin(dim=1)
    return labels


@torch.no_grad()
def kmeans(x, k, *, num_iters=20, batch_size=8192, seed=42, verbose=False):
    if x.shape[0] == 0:
        raise ValueError("kmeans received an empty tensor")
    k = min(int(k), int(x.shape[0]))
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    perm = torch.randperm(x.shape[0], device=x.device, generator=gen)
    centroids = x[perm[:k]].clone()

    for it in range(num_iters):
        labels = assign_labels(x, centroids, batch_size=batch_size)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.bincount(labels, minlength=k).to(x.device)
        new_centroids.index_add_(0, labels, x)

        empty = counts == 0
        if empty.any():
            repl = torch.randint(0, x.shape[0], (int(empty.sum()),), device=x.device, generator=gen)
            new_centroids[empty] = x[repl]
            counts = counts.clone()
            counts[empty] = 1

        new_centroids = new_centroids / counts.clamp_min(1).float().unsqueeze(1)
        shift = (centroids - new_centroids).norm(dim=1).max().item()
        centroids = new_centroids
        if verbose:
            print(f"iter={it:02d} shift={shift:.6f}", flush=True)
        if shift < 1e-4:
            break

    labels = assign_labels(x, centroids, batch_size=batch_size)
    return labels, centroids


def dense_int_labels(values):
    keys = sorted({v for v in values if v is not None}, key=str)
    mapping = {key: idx + 1 for idx, key in enumerate(keys)}
    return [mapping.get(v, 0) for v in values], mapping


def mean_vectors_by_key(keys, vectors):
    groups = defaultdict(list)
    for idx, key in enumerate(keys):
        if key is not None:
            groups[key].append(idx)
    out_keys = list(groups)
    out_vecs = torch.stack([vectors[groups[key]].mean(dim=0) for key in out_keys])
    out_vecs = F.normalize(out_vecs, dim=-1)
    return out_keys, out_vecs, groups


def load_items(cfg):
    data_dir = require(cfg, "data_dir")
    metadata_cfg = cfg.get("metadata", {})
    artist_col = metadata_cfg.get("artist_col", "artist_id")
    album_col = metadata_cfg.get("album_col", "album_id")

    train_items = pl.read_parquet(os.path.join(data_dir, "dvae_train_items.parquet"))
    holdout_items = pl.read_parquet(os.path.join(data_dir, "dvae_holdout_items.parquet"))
    metadata = pl.read_parquet(os.path.join(data_dir, "item_metadata.parquet"))

    items = (
        pl.concat([train_items, holdout_items], how="vertical", rechunk=True)
        .select("item_id", "embed")
        .join(metadata.select("item_id", artist_col, album_col), on="item_id", how="left")
    )

    item_ids = items["item_id"].to_list()
    artist_ids = items[artist_col].to_list()
    album_ids = items[album_col].to_list()
    x_np = np.stack(items["embed"].to_list()).astype("float32")
    x = torch.from_numpy(x_np).to(require(cfg, "device"))
    x = F.normalize(x, dim=-1)

    artist_label, artist_label_map = dense_int_labels(artist_ids)
    album_label, album_label_map = dense_int_labels(album_ids)
    label_metadata_path = os.path.join(
        data_dir,
        metadata_cfg.get("label_metadata_filename", "item_metadata_raw_dense_labels.parquet"),
    )
    pl.DataFrame(
        {
            "item_id": item_ids,
            "artist_label": artist_label,
            "album_label": album_label,
        }
    ).write_parquet(label_metadata_path)

    print("items:", len(item_ids), "dim:", x.shape[-1], flush=True)
    print("raw artist labels:", len(artist_label_map), "raw album labels:", len(album_label_map), flush=True)
    print("missing artists:", sum(v is None for v in artist_ids), flush=True)
    print("missing albums:", sum(v is None for v in album_ids), flush=True)

    return item_ids, artist_ids, album_ids, x, label_metadata_path


@torch.no_grad()
def build_hierarchical_sids(exp, item_ids, artist_ids, album_ids, x):
    seed = int(exp["seed"])
    batch_size = int(exp["batch_size"])
    num_iters = int(exp["kmeans_iters"])

    artist_keys, artist_vecs, _ = mean_vectors_by_key(artist_ids, x)
    artist_labels_raw, artist_centroids_raw = kmeans(
        artist_vecs,
        exp["artist_k"],
        num_iters=num_iters,
        batch_size=batch_size,
        seed=seed,
        verbose=True,
    )
    raw_artist_to_cluster = {
        key: int(label.item()) + 1
        for key, label in zip(artist_keys, artist_labels_raw, strict=True)
    }
    artist_centroids = torch.cat(
        [torch.zeros(1, x.shape[1], device=x.device), artist_centroids_raw],
        dim=0,
    )
    artist_code = torch.tensor(
        [raw_artist_to_cluster.get(a, 0) for a in artist_ids],
        dtype=torch.long,
        device=x.device,
    )

    album_groups = defaultdict(list)
    for idx, (artist_cluster, album_id) in enumerate(zip(artist_code.tolist(), album_ids, strict=True)):
        if album_id is not None:
            album_groups[(artist_cluster, album_id)].append(idx)

    albums_by_artist_cluster = defaultdict(list)
    album_vec_by_key = {}
    for key, idxs in album_groups.items():
        artist_cluster, _ = key
        album_vec = F.normalize(x[idxs].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        album_vec_by_key[key] = album_vec
        albums_by_artist_cluster[artist_cluster].append(key)

    album_code_by_key = {}
    album_centroid_by_local_key = {}
    for artist_cluster, keys in sorted(albums_by_artist_cluster.items()):
        vecs = torch.stack([album_vec_by_key[key] for key in keys])
        labels, centroids = kmeans(
            vecs,
            exp["album_k"],
            num_iters=num_iters,
            batch_size=batch_size,
            seed=seed + int(artist_cluster),
        )
        for key, label in zip(keys, labels.tolist(), strict=True):
            album_code_by_key[key] = int(label) + 1
        for local_code, centroid in enumerate(centroids, start=1):
            album_centroid_by_local_key[(artist_cluster, local_code)] = centroid

    album_code_list = []
    album_centroid_list = []
    zero = torch.zeros(x.shape[1], device=x.device)
    for artist_cluster, album_id in zip(artist_code.tolist(), album_ids, strict=True):
        local_code = album_code_by_key.get((artist_cluster, album_id), 0)
        album_code_list.append(local_code)
        album_centroid_list.append(album_centroid_by_local_key.get((artist_cluster, local_code), zero))

    album_code = torch.tensor(album_code_list, dtype=torch.long, device=x.device)
    album_centroids_per_item = torch.stack(album_centroid_list)

    residual = x - artist_centroids[artist_code] - album_centroids_per_item
    residual_codes = []
    for level in range(int(exp["residual_levels"])):
        labels, centroids = kmeans(
            residual,
            exp["residual_k"],
            num_iters=num_iters,
            batch_size=batch_size,
            seed=seed + 1000 + level,
            verbose=True,
        )
        residual_codes.append(labels)
        residual = residual - centroids[labels]

    codes = torch.stack([artist_code, album_code, *residual_codes], dim=1).cpu().numpy().astype("int64")
    sid_lists = [row.tolist() for row in codes]
    length = codes.shape[1]
    sids = pl.DataFrame(
        {
            "item_id": item_ids,
            "sid": sid_lists,
            "length": [length] * len(item_ids),
        }
    )
    metrics = {
        "name": exp["name"],
        "artist_k_requested": exp["artist_k"],
        "artist_k_effective": int(artist_centroids_raw.shape[0]),
        "album_k_per_artist_requested": exp["album_k"],
        "residual_levels": exp["residual_levels"],
        "residual_k": exp["residual_k"],
        "sid_length": int(length),
        "num_items": len(item_ids),
        "num_raw_artists": len(artist_keys),
        "num_raw_album_artist_pairs": len(album_groups),
    }
    return sids, metrics


def run_cmd(args):
    args = [str(x) for x in args]
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def run_prefix_purity(out_dir, sids_path, label_metadata_path):
    run_cmd(
        [
            sys.executable,
            "-m",
            "scripts.RQ_album_artist_anchor.analyze_prefix_purity",
            "--sids",
            sids_path,
            "--metadata",
            label_metadata_path,
            "--output",
            os.path.join(out_dir, "prefix_purity.json"),
            "--artist-col",
            "artist_label",
            "--album-col",
            "album_label",
        ]
    )


def run_seqrec(cfg, exp):
    base_path = require(cfg, "seqrec_base_config")
    with open(base_path, "r", encoding="utf-8") as f:
        seqrec_cfg = yaml.safe_load(f)

    name = require(exp, "name")
    seqrec_cfg = copy.deepcopy(seqrec_cfg)
    method_dir = os.path.join(require(cfg, "results_dir"), name)
    seqrec_cfg["semantic_ids_path"] = os.path.join(method_dir, "sids.parquet")
    seqrec_cfg["summary_json"] = os.path.join(method_dir, "seqrec_summary.json")

    cfg_path = exp.get("seqrec_config")
    if cfg_path is None:
        cfg_path = f"./configs/RQ_album_artist_anchor/seqrec_{name}.yaml"
    os.makedirs(os.path.dirname(cfg_path) or ".", exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(seqrec_cfg, f, sort_keys=False)

    run_cmd([sys.executable, "-m", "scripts.train_seqrec", "--config", cfg_path])


def main(cfg):
    configure_torch()
    item_ids, artist_ids, album_ids, x, label_metadata_path = load_items(cfg)
    results_dir = require(cfg, "results_dir")

    for exp in require(cfg, "experiments"):
        name = require(exp, "name")
        out_dir = os.path.join(results_dir, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {name} ===", flush=True)

        sids, sid_metrics = build_hierarchical_sids(exp, item_ids, artist_ids, album_ids, x)
        sids_path = os.path.join(out_dir, "sids.parquet")
        sids.write_parquet(sids_path)
        with open(os.path.join(out_dir, "sid_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(sid_metrics, f, indent=2)
        print(json.dumps(sid_metrics, indent=2), flush=True)

        run_prefix_purity(out_dir, sids_path, label_metadata_path)
        if cfg.get("run_seqrec", True):
            run_seqrec(cfg, exp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/RQ_album_artist_anchor/hierarchical_metadata_rkmeans.yaml")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        main(yaml.safe_load(f))
