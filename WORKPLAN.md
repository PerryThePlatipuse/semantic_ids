# WORKPLAN.md

## Goal

Build a small reproducible experiment for artist/album-aware semantic IDs on Yambda.

Main comparison:

```text
Fixed dVAE
vs Original VarLen dVAE
vs Aux artist/album loss
vs Prefix artist/album supervision
```

Metadata-supervision comparison:

```text
Original VarLen dVAE
vs Aux artist/album loss
vs Prefix artist/album supervision
```

Target runtime: about 20 hours on one A100.

## Phase 0 — Setup and tiny run

Deliverables:

* Repo installs.
* Original preprocessing runs.
* Tiny Yambda subset created.
* One tiny VarLen dVAE run completes.
* One tiny seqrec run completes.

Acceptance check:

```bash
python -m scripts.train_dvae --config configs/RQ_album_artist_anchor/original_varlen_dvae_tiny.yaml
python -m scripts.train_seqrec --config configs/RQ_album_artist_anchor/seqrec_original_tiny.yaml
```

## Phase 1 — Yambda subset and metadata

Deliverables:

* `scripts/RQ_album_artist_anchor/build_yambda_subset.py`
* `scripts/RQ_album_artist_anchor/build_artist_album_metadata.py`
* Item table with:

```text
item_id
embed
artist_id or artist_cluster_id
album_id or album_cluster_id
```

Recommended subset:

```text
around 200k users
around 20M interactions
max_core_items around 67k
temporal split preserved
```

This is about 4x smaller than the full Yambda/RQ2 setup by users,
interactions, and train+holdout SID item catalog size.

Notebook:

```text
notebooks/00_data_checks.ipynb
```

Use it to inspect counts, missing metadata, users, items, and split sizes.

## Phase 2 — Original baseline

Deliverables:

* `configs/RQ_album_artist_anchor/fixed_dvae.yaml`
* `configs/RQ_album_artist_anchor/original_varlen_dvae.yaml`
* `configs/RQ_album_artist_anchor/seqrec_fixed.yaml`
* `configs/RQ_album_artist_anchor/seqrec_original.yaml`
* `results/RQ_album_artist_anchor/fixed/sids.parquet`
* `results/RQ_album_artist_anchor/original/sids.parquet`
* `results/RQ_album_artist_anchor/fixed/seqrec_summary.json`
* `results/RQ_album_artist_anchor/original/seqrec_summary.json`

Recommended settings:

```yaml
num_epochs: 3
varlen: true
vocab_size: 4096
maxlen: 5
history_budget: 128 or 256
seqrec depth: 4
beam_size: 50
k_list: [10, 50, 100]
```

## Phase 3 — Aux artist/album loss

Implement auxiliary supervision inside VarLen dVAE.

Possible loss:

```text
loss = recon_loss + beta * KL + lambda_artist * CE(artist_head(z), artist_label)
                         + lambda_album * CE(album_head(z), album_label)
```

Use artist/album clusters if raw class count is too large.

Deliverables:

* Config: `aux_artist_album_loss.yaml`
* Output: `results/RQ_album_artist_anchor/aux/sids.parquet`
* Metrics: `dvae_metrics.json`, `seqrec_summary.json`
* Diagnostics: artist/album aux accuracy if cheap.

Suggested weights:

```text
lambda_artist: 0.01, 0.03, or 0.05
lambda_album: 0.01, 0.03, or 0.05
```

Start with one setting only.

## Phase 4 — Prefix artist/album supervision

Implement soft prefix supervision.

Possible loss:

```text
step 1 / prefix representation predicts artist cluster
step 2 / prefix representation predicts album cluster
```

Avoid hard-coding raw artist or album IDs as mandatory SID tokens unless used only as a fallback baseline.

Deliverables:

* Config: `prefix_artist_album_loss.yaml`
* Output: `results/RQ_album_artist_anchor/prefix/sids.parquet`
* Metrics: `dvae_metrics.json`, `seqrec_summary.json` if time allows.
* Structural diagnostics are mandatory.

## Phase 5 — Analysis

Create:

```text
scripts/RQ_album_artist_anchor/analyze_prefix_purity.py
scripts/RQ_album_artist_anchor/collect_results.py
notebooks/02_analyze_results.ipynb
```

Required diagnostics:

```text
Artist Prefix Purity@1..L
Album Prefix Purity@1..L
mean SID length
collision count
mean items per SID
coverage
Recall/NDCG from seqrec
```

Core result table:

```text
Fixed
Original
AuxLoss
PrefixLoss
```

## Phase 6 — Report and presentation material

Prepare:

```text
report/report.pdf
slides/slides.pdf or slides.pptx
README.md
```

README must include exact commands or notebook entry point.

Recommended reproducibility path:

```bash
python scripts/RQ_album_artist_anchor/build_yambda_subset.py
python scripts/RQ_album_artist_anchor/build_artist_album_metadata.py
python scripts/RQ_album_artist_anchor/run_experiment.py --method fixed
python scripts/RQ_album_artist_anchor/run_experiment.py --method original
python scripts/RQ_album_artist_anchor/run_experiment.py --method aux
python scripts/RQ_album_artist_anchor/run_experiment.py --method prefix
python scripts/RQ_album_artist_anchor/collect_results.py
```

Notebook path is allowed:

```text
notebooks/01_run_experiments.ipynb
```

but it should call the same scripts/configs.

## Time budget

Planned A100 time:

```text
Tiny run:              1–2 h
Original baseline:     3–5 h
AuxLoss:               4–6 h
PrefixLoss:            4–6 h
Analysis/eval buffer:  2–3 h
```

Total target:

```text
14–20 h
```

## Decision rules

If time is low:

1. Keep `Original` and `AuxLoss` end-to-end.
2. Keep `Fixed` as at least a semantic-ID plus structural baseline.
3. Run `PrefixLoss` at least for semantic-ID diagnostics.
4. Skip extra seeds.
5. Skip combined `AuxLoss + PrefixLoss`.
6. Do not run REINFORCE or R-KMeans unless everything else is done.
