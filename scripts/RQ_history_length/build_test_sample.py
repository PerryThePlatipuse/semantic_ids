#!/usr/bin/env python3
import argparse
import os

import polars as pl


def main(args):
    src_path = os.path.join(args.data_dir, args.input)
    dst_path = os.path.join(args.data_dir, args.output)
    test = pl.read_parquet(src_path)
    if args.user_col not in test.columns:
        raise ValueError(f"{src_path} must contain {args.user_col!r}")

    sampled_users = (
        test.select(args.user_col)
        .unique()
        .sample(fraction=args.user_fraction, shuffle=True, seed=args.seed)
    )
    sampled = test.join(sampled_users, on=args.user_col, how="semi")
    sampled.write_parquet(dst_path)
    print(f"Saved {dst_path}: {sampled.height} rows, {sampled[args.user_col].n_unique()} users")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--input", default="seqrec_test_interactions.parquet")
    ap.add_argument("--output", default="seqrec_test_sample_interactions.parquet")
    ap.add_argument("--user-col", default="user_id")
    ap.add_argument("--user-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
