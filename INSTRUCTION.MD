# Instructions

Run commands from the project root:

```bash
cd /home/jupyter/project/semantic_ids
```

## RQ1: Artist/Album Supervision

Prepare Yambda data:

```bash
python3 -m scripts.RQ_album_artist_anchor.build_yambda_subset
python3 -m scripts.RQ_album_artist_anchor.build_artist_album_metadata
```

Run methods:

```bash
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method original --stages dvae,purity,seqrec
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method aux --stages dvae,purity,seqrec
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method prefix --stages dvae,purity,seqrec
```

Collect:

```bash
python3 -m scripts.RQ_album_artist_anchor.collect_results --methods original,aux,prefix
```

## RQ2: Prefix As Dropout

Run prefix-dropout seqrec variants:

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

## RQ3: Amazon Toys Encoder Comparison

Main notebook:

```text
notebooks/04_amazon_toys_encoder_experiment.ipynb
```

Required raw files:

```text
data/Toys_and_Games.jsonl.gz
data/meta_Toys_and_Games.jsonl.gz
```

Set HF token if needed:

```bash
export HF_TOKEN="<token>"
```

The build script reads each `.jsonl.gz` fully into RAM, decompresses in RAM, then writes parquet files:

```bash
python3 -m scripts.RQ_encoder_comparison.build_amazon_toys_embeddings \
  --reviews-gz data/Toys_and_Games.jsonl.gz \
  --meta-gz data/meta_Toys_and_Games.jsonl.gz \
  --output-dir data/amazon_toys_encoders \
  --max-users 5000 \
  --max-interactions 3000000 \
  --batch-size 256 \
  --reuse-prepared \
  --reuse-embeddings
```

Default HF encoders:

```text
tiny_minilm_l3 = sentence-transformers/paraphrase-MiniLM-L3-v2
minilm_l6     = sentence-transformers/all-MiniLM-L6-v2
bge_small     = BAAI/bge-small-en-v1.5
```

Check embeddings:

```bash
python3 -m scripts.RQ_encoder_comparison.check_embeddings \
  --embeddings-dir data/amazon_toys_encoders/embeddings \
  --output results/RQ_encoder_comparison_amazon_toys/embedding_checks.json
```

Smoke test:

```bash
python3 -m scripts.RQ_encoder_comparison.run_encoder_experiment \
  --interactions data/amazon_toys_encoders/interactions.parquet \
  --embeddings-dir data/amazon_toys_encoders/embeddings \
  --work-data-dir data/RQ_encoder_comparison/amazon_toys_smoke \
  --results-dir results/RQ_encoder_comparison_amazon_toys_smoke \
  --dvae-config configs/RQ_encoder_comparison/amazon_toys_fixed_dvae.yaml \
  --seqrec-config configs/RQ_encoder_comparison/seqrec_amazon_toys_fixed.yaml \
  --stages prepare,dvae,sid_metrics \
  --max-users 1000 \
  --max-core-items 5000 \
  --num-epochs 1
```

Full quick run:

```bash
python3 -m scripts.RQ_encoder_comparison.run_encoder_experiment \
  --interactions data/amazon_toys_encoders/interactions.parquet \
  --embeddings-dir data/amazon_toys_encoders/embeddings \
  --work-data-dir data/RQ_encoder_comparison/amazon_toys \
  --results-dir results/RQ_encoder_comparison_amazon_toys \
  --dvae-config configs/RQ_encoder_comparison/amazon_toys_fixed_dvae.yaml \
  --seqrec-config configs/RQ_encoder_comparison/seqrec_amazon_toys_fixed.yaml \
  --stages prepare,dvae,sid_metrics,seqrec \
  --max-users 5000 \
  --max-core-items 30000
```

Collect:

```bash
python3 -m scripts.RQ_encoder_comparison.collect_results \
  --results-dir results/RQ_encoder_comparison_amazon_toys
```

Outputs:

```text
data/amazon_toys_encoders/interactions.parquet
data/amazon_toys_encoders/item_texts.parquet
data/amazon_toys_encoders/embeddings/<encoder>.parquet
results/RQ_encoder_comparison_amazon_toys/<encoder>/dvae_metrics.json
results/RQ_encoder_comparison_amazon_toys/<encoder>/sids.parquet
results/RQ_encoder_comparison_amazon_toys/<encoder>/sid_metrics.json
results/RQ_encoder_comparison_amazon_toys/<encoder>/seqrec_summary.json
results/RQ_encoder_comparison_amazon_toys/comparison.csv
```
