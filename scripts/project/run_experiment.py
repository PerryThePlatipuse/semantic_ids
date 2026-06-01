#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


METHODS = {
    "original": {
        "dvae": "configs/project/original_varlen_dvae.yaml",
        "seqrec": "configs/project/seqrec_original.yaml",
    },
    "aux": {
        "dvae": "configs/project/aux_artist_album_loss.yaml",
        "seqrec": "configs/project/seqrec_aux.yaml",
    },
    "prefix": {
        "dvae": "configs/project/prefix_artist_album_loss.yaml",
        "seqrec": "configs/project/seqrec_prefix.yaml",
    },
}


def _run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main(args):
    method_cfg = METHODS[args.method]
    out_dir = os.path.join("results", "project", args.method)
    stages = args.stages.split(",")
    unknown = set(stages) - {"dvae", "purity", "seqrec"}
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")

    if "dvae" in stages:
        _run(sys.executable, "-m", "scripts.train_dvae", "--config", method_cfg["dvae"])
    if "purity" in stages:
        _run(
            sys.executable,
            "-m",
            "scripts.project.analyze_prefix_purity",
            "--sids",
            os.path.join(out_dir, "sids.parquet"),
            "--output",
            os.path.join(out_dir, "prefix_purity.json"),
        )
    if "seqrec" in stages:
        _run(sys.executable, "-m", "scripts.train_seqrec", "--config", method_cfg["seqrec"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=sorted(METHODS), required=True)
    ap.add_argument("--stages", default="dvae,purity,seqrec")
    main(ap.parse_args())
