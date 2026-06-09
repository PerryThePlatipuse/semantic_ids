# Variable-Length Semantic IDs for Recommender Systems

This is the official repository for the paper “Variable-Length Semantic IDs for Recommender Systems”.
It contains the full implementation of all proposed methods, along with the exact experimental configurations and detailed results required to reproduce the findings reported in the paper.

---

## 1. Environment setup

### 1.1 Install Python dependencies

```bash
pip install -r requirements.txt
```

Some optional acceleration libraries (`xformers`, `triton`) may require a compatible CUDA and compiler setup.

---

### 1.2 Hugging Face authentication (Amazon only)

Preprocessing the Amazon dataset requires downloading pretrained text models and computing EmbeddingGemma embeddings.
Please authenticate with Hugging Face:

```bash
huggingface-cli login
```

---

## 2. Dataset preparation

### 2.1 Downloading datasets

#### Yambda

We provide a helper for downloading user-item interactions and item embeddings:

```python
from scripts.data.yambda import download

download(dst_dir="./data/yambda")
```

This creates:

* `data/yambda/interactions.parquet`
* `data/yambda/embeddings.parquet`

---

#### VK-LSVD

Use the analogous helper function `download` from:

* `scripts/data/vklsvd.py`

---

#### Amazon (Amazon Reviews 2023)

Amazon preprocessing requires a manual download step:

1. Download **metadata** and **reviews** from
   [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)

2. Run the function `process` from the helper script:

* `scripts/data/amazon.py`

The Amazon helper also computes **Gemma embeddings** and therefore requires Hugging Face access.

---

### 2.2 Dataset preprocessing

Each dataset provides a preprocessing script that performs:

* data for training and evaluating both semantic ID construction methods and sequential recommendation models.

For **Yambda**:

```bash
python scripts/data/yambda.py
```

The output is written to the destination directory specified in `main()` (default: `./data/yambda`).

Additionally, the script writes a user subsample of the test interactions to (5\% for Yambda, 10\% for VK-LSVD) for RQ-2:

* `./data/yambda/seqrec_test_sample_interactions.parquet`

---

## 3. Running experiments

### 3.1 Training semantic ID models

We provide separate training scripts for each semantic-ID method:

```bash
python -m scripts.train_X --config path/to/config.yaml
```

where `X` is one of:

* `dvae`
* `reinforce`
* `rkmeans`

---

### 3.2 Training the sequential recommender

The sequential recommender is trained using:

```bash
python -m scripts.train_seqrec --config path/to/config.yaml
```

---

### 3.3 Configurations and outputs

* All configurations used for **RQ1–RQ4** are provided in `configs/`; for **RQ3**, we reuse dVAE results on Yambda from **RQ1**
* Evaluation outputs are stored under `results/`.

Each experiment can be reproduced by running the corresponding training script with the appropriate config file.

---

## 4. Reproducibility notes

* All experiments are fully configuration-driven via YAML files.
* Random seeds are fixed in preprocessing and training scripts.
* Evaluation follows the exact protocols described in the paper.
* We provide actual paper evaluation results under `results/` for reference.

---

## 5. Course project: artist/album-aware Yambda SIDs

The course-project extension compares fixed-length dVAE, original VarLen dVAE,
auxiliary artist/album loss, and prefix-level artist/album supervision. By
default it uses the same full Yambda-scale preprocessing as the paper configs,
with hashed metadata classes and a smaller seqrec transformer.

Download the original Yambda inputs and metadata:

```python
from scripts.data.yambda import download, download_metadata

download(dst_dir="./data/yambda")
download_metadata(dst_dir="./data/yambda")
```

Build the project data and attach artist/album labels:

```bash
python3 -m scripts.RQ_album_artist_anchor.build_yambda_subset
python3 -m scripts.RQ_album_artist_anchor.build_artist_album_metadata
```

By default this keeps all users, interactions, and core items available after
the standard Yambda temporal/core filtering. For a cheaper run, pass explicit
`--num-users`, `--max-interactions`, and `--max-core-items` values.

Before a full run, validate the original pipeline with the tiny configs:

```bash
python3 -m scripts.train_dvae --config configs/RQ_album_artist_anchor/original_varlen_dvae_tiny.yaml
python3 -m scripts.train_seqrec --config configs/RQ_album_artist_anchor/seqrec_original_tiny.yaml
```

Run one method end to end:

```bash
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method fixed
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method original
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method aux
python3 -m scripts.RQ_album_artist_anchor.run_experiment --method prefix
```

Each runner accepts `--stages dvae,purity,seqrec`, so expensive stages can be
rerun separately. Seqrec configs report `@10`, `@50`, and `@100` metrics.
Collect the final comparison table with:

```bash
python3 -m scripts.RQ_album_artist_anchor.collect_results
```

Project configs live under `configs/RQ_album_artist_anchor/`. Outputs are written to
`results/RQ_album_artist_anchor/<method>/`.

### 5.1 Prefix dropout experiment

Prefix dropout is a training-time augmentation for seqrec. It randomly truncates
SID sequences to their first keep codes with probability p, forcing the model
to generalize from coarse prefixes. The dVAE is **not** retrained — both baseline
and dropout runs use the same `prefix/sids.parquet` (supervised SIDs).

Run the baseline (same config but without dropout, `p=0`):

```bash
python3 -m scripts.train_seqrec --config configs/RQ_album_artist_anchor/seqrec_original.yaml
```

Run prefix dropout (`p=0.2, keep=1`):

```bash
python3 -m scripts.RQ_album_artist_anchor.train_seqrec_prefix_dropout \
    --config configs/RQ_album_artist_anchor/seqrec_prefix_dropout.yaml
```

Results are written to `results/RQ_album_artist_anchor/prefix_dropout/`.

