#!/usr/bin/env python3
"""Write HTCondor submit files for the GAE pipeline on the FNAL LPC.

Graph production and scoring are split into one job per shard of a sample's
file list. Training is a single job, optionally on a GPU.

    python condor/submit_condor.py --stage produce --samples background_2018 \
        --files-per-job 5
    python condor/submit_condor.py --stage score --samples signal_2018 \
        --files-per-job 5 --job-arguments "--checkpoint training/run/checkpoint.pt"
    python condor/submit_condor.py --stage train --request-gpus 1
    python condor/submit_condor.py --stage contamination --years 2018 \
        --files-per-job 10

Nothing is submitted automatically. Inspect the generated .jdl and run
condor_submit on it.
"""

import argparse
import os
import subprocess
import sys

SVJ_GNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDOR_DIR = os.path.join(SVJ_GNN_DIR, "condor")
sys.path.insert(0, SVJ_GNN_DIR)

from signal_contamination import contamination_buckets  # noqa: E402
from utils.GaeConfig import load_config  # noqa: E402
from utils.InputJson import shard_files  # noqa: E402


# Directories and files that must not go into the job tarball.
TAR_EXCLUDES = (
    ".git",
    ".venv-gae",
    "data",
    "scores",
    "plots",
    "training",
    "condor/logs",
    "*.ipynb",
    "*.root",
    "__pycache__",
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("produce", "score", "train", "contamination"),
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2018"],
        help="Years for the contamination stage",
    )
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--files-per-job", type=int, default=5)
    parser.add_argument("--submit-dir", default=os.path.join(CONDOR_DIR, "submit"))
    parser.add_argument("--log-dir", default=os.path.join(CONDOR_DIR, "logs"))
    parser.add_argument("--request-cpus", type=int, default=1)
    parser.add_argument("--request-memory", default="4 GB")
    parser.add_argument("--request-disk", default="4 GB")
    parser.add_argument(
        "--request-gpus",
        type=int,
        default=None,
        help="Request GPUs; useful for the train stage",
    )
    parser.add_argument(
        "--proxy",
        default=os.path.join(
            os.environ.get("HOME", ""), "x509up_file"
        ),
        help="Grid proxy to ship with the job, needed to read files over xrootd",
    )
    parser.add_argument(
        "--job-arguments",
        default="",
        help="Extra arguments appended to the stage's python command",
    )
    parser.add_argument("--skip-tarball", action="store_true")
    return parser.parse_args()


def make_tarball(path):
    """Tar the SVJ_GNN directory, leaving out inputs and outputs."""
    excludes = []
    for pattern in TAR_EXCLUDES:
        excludes.extend(["--exclude", pattern])
    command = [
        "tar",
        "-czf",
        path,
        *excludes,
        "-C",
        os.path.dirname(SVJ_GNN_DIR),
        os.path.basename(SVJ_GNN_DIR),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)
    return path


def contamination_queue_items(args):
    items = []
    for year in args.years:
        for sample_label, _bucket, paths in contamination_buckets(
            SVJ_GNN_DIR, year
        ):
            n_shards = len(shard_files(paths, args.files_per_job))
            print(f"Contamination '{sample_label}': {n_shards} shard(s)")
            items.extend(
                f"contamination, {sample_label}, {shard}"
                for shard in range(n_shards)
            )
    return items


def queue_items(args, config):
    """One ``stage, sample, shard`` line per job."""
    if args.stage == "train":
        return [f"{args.stage}, train, 0"]
    if args.stage == "contamination":
        return contamination_queue_items(args)

    items = []
    for sample in config.select(args.samples):
        n_shards = len(shard_files(sample.files(), args.files_per_job))
        print(f"Sample '{sample.name}': {n_shards} shard(s)")
        items.extend(
            f"{args.stage}, {sample.name}, {shard}" for shard in range(n_shards)
        )
    return items


def job_arguments(args):
    """Arguments appended after the stage, sample and shard."""
    if args.stage == "contamination":
        pieces = [f"--files-per-job {args.files_per_job}"]
    elif args.stage == "train":
        pieces = [f"--config {args.config}"]
    else:
        pieces = [f"--config {args.config}"]
        pieces.append(f"--files-per-job {args.files_per_job}")
    if args.job_arguments:
        pieces.append(args.job_arguments)
    return " ".join(pieces)


def main():
    args = parse_arguments()
    if args.stage != "contamination":
        config = load_config(os.path.join(SVJ_GNN_DIR, args.config))
    else:
        config = None

    os.makedirs(args.submit_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    tarball = os.path.join(args.submit_dir, "SVJ_GNN.tar.gz")
    if not args.skip_tarball:
        make_tarball(tarball)

    with open(os.path.join(CONDOR_DIR, "job.jdl.tmpl")) as template_file:
        template = template_file.read()

    gpu_requirement = (
        f"request_gpus           = {args.request_gpus}"
        if args.request_gpus
        else "# no GPU requested"
    )

    submit_file = template
    for placeholder, value in (
        ("CONDOR_DIR", CONDOR_DIR),
        ("TARBALL", tarball),
        ("X509_PROXY", args.proxy),
        ("LOG_DIR", args.log_dir),
        ("REQUEST_CPUS", str(args.request_cpus)),
        ("REQUEST_MEMORY", args.request_memory),
        ("REQUEST_DISK", args.request_disk),
        ("GPU_REQUIREMENT", gpu_requirement),
        ("JOB_ARGUMENTS", job_arguments(args)),
        ("QUEUE_ITEMS", "\n".join(queue_items(args, config))),
    ):
        submit_file = submit_file.replace(placeholder, value)

    submit_path = os.path.join(args.submit_dir, f"{args.stage}.jdl")
    with open(submit_path, "w") as output_file:
        output_file.write(submit_file)

    print(f"Wrote {submit_path}")
    print(f"Submit it with: condor_submit {submit_path}")


if __name__ == "__main__":
    main()
