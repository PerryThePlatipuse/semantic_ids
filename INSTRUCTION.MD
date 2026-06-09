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
