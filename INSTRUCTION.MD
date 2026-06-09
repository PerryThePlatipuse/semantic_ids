# Instructions

Run commands from the project root:

```bash
cd /home/jupyter/project/semantic_ids
```

## Prepare Data

```bash
python3 -m scripts.RQ_album_artist_anchor.build_yambda_subset
python3 -m scripts.RQ_album_artist_anchor.build_artist_album_metadata
```

## RQ1: Artist/Album Supervision

Baseline:

```bash
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method original --stages dvae,purity,seqrec
```

Aux artist/album loss:

```bash
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method aux --stages dvae,purity,seqrec
```

Prefix artist/album loss:

```bash
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method prefix --stages dvae,purity,seqrec
```

Collect comparison:

```bash
python3 -m scripts.RQ_album_artist_anchor.collect_results --methods original,aux,prefix
```

## RQ2: Prefix As Dropout

Run prefix-dropout seqrec with different prefix lengths:

```bash
python3 -m scripts.RQ_album_artist_anchor.train_seqrec_prefix_dropout --config configs/RQ_album_artist_anchor/seqrec_prefix_dropout.yaml
python3 -m scripts.RQ_album_artist_anchor.train_seqrec_prefix_dropout --config configs/RQ_album_artist_anchor/seqrec_prefix_dropout_keep2.yaml
python3 -m scripts.RQ_album_artist_anchor.train_seqrec_prefix_dropout --config configs/RQ_album_artist_anchor/seqrec_prefix_dropout_keep3.yaml
```

Outputs:

```text
results/RQ_album_artist_anchor/prefix_dropout/seqrec_summary.json
results/RQ_album_artist_anchor/prefix_dropout_keep2/seqrec_summary.json
results/RQ_album_artist_anchor/prefix_dropout_keep3/seqrec_summary.json
```

## RQ3: Encoder Comparison On Amazon

Required inputs:

```text
data/amazon/interactions.parquet
data/amazon/embeddings/<encoder_name>.parquet
```

Each embedding parquet should contain:

```text
item_id or parent_asin
embed or embedding
```

Run one or more encoders:

```bash
python3 -m scripts.RQ_encoder_comparison.run_encoder_experiment \
  --interactions data/amazon/interactions.parquet \
  --embedding encoder_a=data/amazon/embeddings/encoder_a.parquet \
  --embedding encoder_b=data/amazon/embeddings/encoder_b.parquet
```

Collect comparison:

```bash
python3 -m scripts.RQ_encoder_comparison.collect_results
```

Outputs:

```text
results/RQ_encoder_comparison/<encoder>/dvae_metrics.json
results/RQ_encoder_comparison/<encoder>/sids.parquet
results/RQ_encoder_comparison/<encoder>/sid_metrics.json
results/RQ_encoder_comparison/<encoder>/seqrec_summary.json
results/RQ_encoder_comparison/comparison.csv
```

## RQ3 Quick Yambda E5 Notebook

Use this notebook when you have teammate-provided E5 embeddings:

```text
notebooks/04_yambda_e5_encoder_experiment.ipynb
```

Expected file:

```text
data/yambda_e5_raw/embeddings_e5.parquet
```

The notebook will reuse old Yambda interactions if available, otherwise it downloads `yandex/yambda` likes.

Old interactions are enough if one of these exists:

```text
data/yambda/interactions.parquet
data/RQ_album_artist_anchor/yambda/subset_interactions.parquet
data/yambda_e5_raw/interactions.parquet
```

Old prepared files like `dvae_train_items.parquet` are not enough for a new encoder, because they contain old embeddings. Rebuild prepared data with the new embedding parquet.

Smoke test:

```bash
python3 -m scripts.RQ_encoder_comparison.run_encoder_experiment \
  --interactions data/yambda_e5_raw/interactions.parquet \
  --embedding e5=data/yambda_e5_raw/embeddings_e5.parquet \
  --test-interval-days 7 \
  --work-data-dir data/RQ_encoder_comparison/yambda \
  --results-dir results/RQ_encoder_comparison_yambda \
  --stages prepare,dvae,sid_metrics \
  --max-users 1000 \
  --max-core-items 5000 \
  --num-epochs 1
```

Full quick run:

```bash
python3 -m scripts.RQ_encoder_comparison.run_encoder_experiment \
  --interactions data/yambda_e5_raw/interactions.parquet \
  --embedding e5=data/yambda_e5_raw/embeddings_e5.parquet \
  --test-interval-days 7 \
  --work-data-dir data/RQ_encoder_comparison/yambda \
  --results-dir results/RQ_encoder_comparison_yambda \
  --stages prepare,dvae,sid_metrics,seqrec \
  --max-users 5000 \
  --max-core-items 30000 \
  --num-epochs 3
```

Collect:

```bash
python3 -m scripts.RQ_encoder_comparison.collect_results \
  --results-dir results/RQ_encoder_comparison_yambda
```
