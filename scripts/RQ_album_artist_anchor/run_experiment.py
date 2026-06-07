#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


PURITY_METADATA = "data/RQ_album_artist_anchor/yambda/item_metadata_raw_dense_labels.parquet"


METHODS = {
    "fixed": {
        "dvae": "configs/RQ_album_artist_anchor/fixed_dvae.yaml",
        "seqrec": "configs/RQ_album_artist_anchor/seqrec_fixed.yaml",
    },
    "original": {
        "dvae": "configs/RQ_album_artist_anchor/original_varlen_dvae.yaml",
        "seqrec": "configs/RQ_album_artist_anchor/seqrec_original.yaml",
    },
    "aux": {
        "dvae": "configs/RQ_album_artist_anchor/aux_artist_album_loss.yaml",
        "seqrec": "configs/RQ_album_artist_anchor/seqrec_aux.yaml",
    },
    "prefix": {
        "dvae": "configs/RQ_album_artist_anchor/prefix_artist_album_loss.yaml",
        "seqrec": "configs/RQ_album_artist_anchor/seqrec_prefix.yaml",
    },
    "rkmeans": {
        "rkmeans": "configs/RQ_album_artist_anchor/rkmeans.yaml",
        "seqrec": "configs/RQ_album_artist_anchor/seqrec_rkmeans.yaml",
    },
}


def _run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main(args):
    method_cfg = METHODS[args.method]
    out_dir = os.path.join("results", "RQ_album_artist_anchor", args.method)
    stages = args.stages.split(",")
    unknown = set(stages) - {"dvae", "rkmeans", "purity", "seqrec"}
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    if "dvae" in stages and "dvae" in method_cfg:
        _run(sys.executable, "-m", "scripts.train_dvae", "--config", method_cfg["dvae"])
    if ("rkmeans" in stages or "dvae" in stages) and "rkmeans" in method_cfg:
        _run(sys.executable, "-m", "scripts.train_rkmeans", "--config", method_cfg["rkmeans"])
    if "purity" in stages:
        _run(
            sys.executable,
            "-m",
            "scripts.RQ_album_artist_anchor.analyze_prefix_purity",
            "--sids",
            os.path.join(out_dir, "sids.parquet"),
            "--metadata",
            PURITY_METADATA,
            "--output",
            os.path.join(out_dir, "prefix_purity.json"),
            "--artist-col",
            "artist_label",
            "--album-col",
            "album_label",
        )
    if "seqrec" in stages:
        _run(sys.executable, "-m", "scripts.train_seqrec", "--config", method_cfg["seqrec"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--stages", default="dvae,purity,seqrec")
    main(ap.parse_args())
