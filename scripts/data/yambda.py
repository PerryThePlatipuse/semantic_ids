import os
import tqdm

import numpy as np
import polars as pl

from .utils import preprocess_data
from scripts.data.bge import build_embeddings
from scripts.data.e5 import build_item_text, encode_e5


TEST_INTERVAL = 7 * 24 * 60 * 60 # one week


def download(dst_dir, encoder=None):
    from datasets import load_dataset

    os.makedirs(dst_dir, exist_ok=True)
    ds = load_dataset("yandex/yambda", data_dir="flat/5b", data_files="likes.parquet")
    ds = ds['train'].to_polars()
    ds.write_parquet(os.path.join(dst_dir, 'interactions.parquet'))

    if encoder == "bge":
        
        artist = load_dataset("yandex/yambda", data_files="artist_item_mapping.parquet")["train"].to_polars()
        album = load_dataset("yandex/yambda", data_files="album_item_mapping.parquet")["train"].to_polars()

        liked_items = ds.select("item_id").unique()

        embeddings = build_embeddings(
            item_ids=liked_items,
            artist_map=artist,
            album_map=album
        )

        embeddings_bge = embeddings.with_columns(
            pl.col("embed").cast(pl.Array(pl.Float32, (1024,)))
        )

        embeddings_bge.write_parquet(os.path.join(dst_dir, "embeddings_bge.parquet"))
        return ds, embeddings_bge
    
    elif encoder == "e5":

        artist = load_dataset("yandex/yambda", data_files="artist_item_mapping.parquet")["train"].to_polars()
        album = load_dataset("yandex/yambda", data_files="album_item_mapping.parquet")["train"].to_polars()

        liked_items = ds.select("item_id").unique()
        item_ids, texts = build_item_text(liked_items, artist, album)
        emb = encode_e5(texts)

        embeddings_e5 = pl.DataFrame({
            "item_id": item_ids,
            "embed": list(emb)
        }).with_columns(
            pl.col("embed").cast(pl.Array(pl.Float32, (1024,)))
        )

        embeddings_e5.write_parquet(os.path.join(dst_dir, "embeddings_e5.parquet"))
        return ds, embeddings_e5

    embeddings = load_dataset("yandex/yambda", data_dir="", data_files="embeddings.parquet")
    embeddings = embeddings['train'].to_polars()

    liked_items = ds.select('item_id').unique()
    filtered_embeddings = embeddings \
        .join(liked_items, on='item_id', how='semi') \
        .select('item_id', pl.col('embed').cast(pl.Array(pl.Float32, (128,))))
    filtered_embeddings.write_parquet(os.path.join(dst_dir, 'embeddings.parquet'))
    
    return ds, filtered_embeddings


def download_metadata(dst_dir):
    from datasets import load_dataset

    os.makedirs(dst_dir, exist_ok=True)
    outputs = {}
    for name in ("artist_item_mapping", "album_item_mapping"):
        ds = load_dataset("yandex/yambda", data_dir="", data_files=f"{name}.parquet")
        df = ds["train"].to_polars()
        df.write_parquet(os.path.join(dst_dir, f"{name}.parquet"))
        outputs[name] = df
    return outputs


def main(data_dir, dst_dir, core_threshold=16, holdout_frac=0.1, seed=42, topk_head=30000):
    item_embeddings = pl.read_parquet(os.path.join(data_dir, 'embeddings.parquet'))
    
    interactions = pl.read_parquet(os.path.join(data_dir, 'interactions.parquet')) \
        .join(item_embeddings, on='item_id', how='semi') \
        .rename({'uid': 'user_id'})

    max_timestamp = interactions['timestamp'].max()
    train = interactions.filter(pl.col('timestamp') < max_timestamp - TEST_INTERVAL)
    test = interactions.filter(pl.col('timestamp') >= max_timestamp - TEST_INTERVAL)

    preprocess_data(train, test, item_embeddings, dst_dir, core_threshold, holdout_frac, seed, topk_head=topk_head, verbose=True)


if __name__ == '__main__':
    main(
        data_dir='../data/yambda/', 
        dst_dir='./data/yambda', 
        core_threshold=16,
        holdout_frac=0.1,
        seed=42,
        topk_head=30000
    )
    
    seqrec_test_interactions = pl.read_parquet('./data/yambda/seqrec_test_interactions.parquet')
    sampled_users = seqrec_test_interactions.select('user_id') \
        .unique().sample(fraction=0.05, shuffle=True, seed=42)\

    seqrec_test_interactions \
        .join(sampled_users, on='user_id', how='semi') \
        .write_parquet('./data/yambda/seqrec_test_sample_interactions.parquet')
