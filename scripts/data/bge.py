import polars as pl
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-large-en-v1.5"
model = SentenceTransformer(MODEL_NAME)


def build_item_text(items: pl.DataFrame,
                    artist_map: pl.DataFrame = None,
                    album_map: pl.DataFrame = None):

    df = items.unique(subset=["item_id"])

    if artist_map is not None:
        artist_map = artist_map.unique(subset=["item_id"])
        df = df.join(artist_map, on="item_id", how="left")

    if album_map is not None:
        album_map = album_map.unique(subset=["item_id"])
        df = df.join(album_map, on="item_id", how="left")

    def safe_str(x):
        return "" if x is None else str(x)

    texts = [
        " | ".join([
            safe_str(row.get("artist_id")),
            safe_str(row.get("album_id")),
        ]).strip() or "unknown item"
        for row in df.iter_rows(named=True)
    ]

    item_ids = df["item_id"].to_list()

    return item_ids, texts


def encode_bge(texts, batch_size=256, device="cuda"):

    emb = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return emb


def build_embeddings(item_ids: pl.DataFrame,
                     artist_map: pl.DataFrame,
                     album_map: pl.DataFrame):

    item_ids_aligned, texts = build_item_text(item_ids, artist_map, album_map)

    emb = encode_bge(texts)

    assert len(item_ids_aligned) == len(emb), "alignment broken"

    return pl.DataFrame({
        "item_id": item_ids_aligned,
        "embed": list(emb)
    })
