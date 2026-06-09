#!/usr/bin/env python3
import os
import json
import argparse

import numpy as np
import polars as pl
import yaml

from scripts.train_seqrec import (
    build_dataframes_and_stats,
    run_train,
    run_eval,
)
from lib.utils import configure_torch


def apply_prefix_dropout(pretrain_df, stats, cfg):
    """
    Args:
        pretrain_df: DataFrame с колонками [uid, token_id: list[UInt32]]
        stats: словарь со статистиками (unique_token_offset, etc.)
        cfg: конфиг с prefix_dropout_prob и prefix_dropout_keep

    Returns: Модифицированный pretrain_df
    """
    p = float(cfg.get("prefix_dropout_prob", 0.0))
    keep = int(cfg.get("prefix_dropout_keep", 1))
    seed = int(cfg.get("seed", 42))

    if p <= 0.0:
        return pretrain_df

    unique_offset = stats.get("unique_token_offset")
    append_unique = stats.get("append_unique_token", True)

    if not append_unique or unique_offset is None:
        return pretrain_df

    # юзаем другой seed чтобы не коррелировать с шаффлом данных
    rng = np.random.default_rng(seed + 777)

    def dropout_stream(tokens):
        """
        обрабатываем один поток токенов - одного юзера

        проходим по потоку, находит границы треков по unique токенам,
        и с вероятностью p обрезает трек до первых keep кодов
        """
        result = [tokens[0]]  # user_start  всегда оставляем
        i = 1
        while i < len(tokens):
            track_start = i
            while i < len(tokens) and tokens[i] < unique_offset:
                i += 1
            codes = tokens[track_start:i]
            has_unique = (i < len(tokens))

            if rng.random() < p and len(codes) > keep:
                result.extend(codes[:keep])
            else:
                result.extend(codes)
                if has_unique:
                    result.append(tokens[i])

            if has_unique:
                i += 1  

        return result

    streams = pretrain_df["token_id"].to_list()
    n_before = sum(len(s) for s in streams)

    dropped_streams = [dropout_stream(s) for s in streams]
    n_after = sum(len(s) for s in dropped_streams)

    n_users = len(streams)
    print(f"Prefix dropout: p={p}, keep={keep}")
    print(f"пользователей:{n_users:,}")
    print(f"токенов до:{n_before:,}")
    print(f"токенов после:{n_after:,}")
    print(f"выброшено:{n_before - n_after:,} ({(1 - n_after / n_before) * 100:.1f}%)")

    return pretrain_df.with_columns(
        pl.Series("token_id", dropped_streams, dtype=pl.List(pl.UInt32))
    )


def main(cfg):
    configure_torch()

    pretrain_df, test_df, processed_sids, head_items_df, targets_df, history_df, stats = \
        build_dataframes_and_stats(cfg)

    pretrain_df = apply_prefix_dropout(pretrain_df, stats, cfg)

    if pretrain_df.height > 0:
        new_max = int(pretrain_df["token_id"].list.len().max())
        old_max = stats["pretrain_max_len_observed"]
        stats["pretrain_max_len_observed"] = new_max
        if new_max != old_max:
            print(f"max_seq_len: {old_max} -> {new_max}")


    graph = run_train(cfg, stats, pretrain_df)

 
    metrics = run_eval(
        cfg, stats, graph,
        processed_semantic_ids=processed_sids,
        test_df=test_df,
        targets_df=targets_df,
        head_items_df=head_items_df,
        history_df=history_df,
    )

    result = {"metrics": metrics, "stats": stats, "cfg": cfg}

    summary_path = cfg.get("summary_json")
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=True)
        
    for key in ["recall@10", "recall@50", "ndcg@10", "ndcg@50",
                 "tail_recall@10", "tail_recall@50", "long_tail_share",
                 "coverage@10"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Train seqrec with prefix dropout"
    )
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
