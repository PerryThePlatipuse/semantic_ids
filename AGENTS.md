# AGENTS.md

## Project context

We extend `KhrylchenkoKirill/varlen_semantic_ids` for a course project on semantic IDs for music recommendation.

Main RQ:

> Can artist/album supervision improve variable-length semantic IDs for music recommendation?

We focus on Yambda and compare three variants:

1. `original_varlen_dvae`: unchanged baseline from the repository.
2. `aux_artist_album_loss`: VarLen dVAE with auxiliary artist/album prediction or reconstruction loss.
3. `prefix_artist_album_loss`: VarLen dVAE with prefix-level artist/album supervision.

The goal is not to reproduce the full paper. The goal is a small, clean, reproducible experiment under limited compute.

## Compute target

Target budget: about 20 hours on one A100.

Prefer:

* Yambda subset, not full-scale run.
* One seed.
* One main semantic-ID model: VarLen dVAE.
* Lighter seqrec config than paper defaults.

Recommended reductions:

* 2k–5k users.
* 1M–3M interactions.
* `history_budget: 128` or `256`.
* seqrec `depth: 4`.
* eval `beam_size: 20` or `50`.
* dVAE `num_epochs: 3` for first full runs.

## Implementation rules

Keep changes small and config-driven.

Preferred new files:

```text
notebooks/
  00_data_checks.ipynb
  01_run_experiments.ipynb
  02_analyze_results.ipynb

configs/RQ_album_artist_anchor/
  original_varlen_dvae.yaml
  aux_artist_album_loss.yaml
  prefix_artist_album_loss.yaml
  seqrec_original.yaml
  seqrec_aux.yaml
  seqrec_prefix.yaml

scripts/RQ_album_artist_anchor/
  build_yambda_subset.py
  build_artist_album_metadata.py
  run_experiment.py
  analyze_prefix_purity.py
  collect_results.py
```

Do not make notebooks the only reproducibility path. Notebooks may call scripts, generate configs, inspect outputs, and create report tables.

## Required outputs

Each method should produce:

```text
results/RQ_album_artist_anchor/<method>/sids.parquet
results/RQ_album_artist_anchor/<method>/dvae_metrics.json
results/RQ_album_artist_anchor/<method>/seqrec_summary.json
results/RQ_album_artist_anchor/<method>/prefix_purity.json
```

Final comparison should include:

* Recall@10 / Recall@100 or Recall@50.
* NDCG@10 / NDCG@100 or NDCG@50.
* Coverage.
* Head/tail metrics if available.
* Artist prefix purity.
* Album prefix purity.
* Collision or bucket-size statistics.

## Fallback policy

If full seqrec for all methods is too slow:

1. Run seqrec for `original_varlen_dvae` and `aux_artist_album_loss`.
2. For `prefix_artist_album_loss`, report structural metrics: prefix purity, collisions, mean SID length.
3. Clearly state the compute limitation.

If raw artist/album classes are too many, use clustered or hashed labels:

```text
artist_cluster_id: 256–1024 classes
album_cluster_id: 512–2048 classes
```

Prefer a working simple method over a broken clever method.
