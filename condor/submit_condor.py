#!/usr/bin/env python3
"""Write HTCondor submit files for the GAE pipeline on the FNAL LPC.

Graph production and scoring are split into one job per shard of a sample's
file list. Training is a single job, optionally on a GPU.

    python condor/submit_condor.py --stage produce --samples background_2018 \
        --files-per-job 5
    python condor/submit_condor.py --stage score --samples signal_2018 \
        --files-per-job 5 --job-arguments "--checkpoint training/run/checkpoint.pt"
    python condor/submit_condor.py --stage train

Nothing is submitted automatically. Inspect the generated .jdl and run
condor_submit on it.
"""

import argparse
import os
import shutil
import subprocess
import sys

SVJ_GNN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONDOR_DIR = os.path.join(SVJ_GNN_DIR, "condor")
sys.path.insert(0, SVJ_GNN_DIR)

from utils.GaeConfig import load_config  # noqa: E402
from utils.InputJson import shard_files  # noqa: E402


# Directories and files that must not go into the job tarball.
TAR_EXCLUDES = (
    ".git",
    ".venv-gae",
    "data",
    "scores",
    "output_scores",
    "plots",
    "training",
    "condor/logs",
    "*.ipynb",
    "*.root",
    "__pycache__",
)


def resolve_and_prepare_proxy(proxy_arg=None):
    """Ensure a valid X.509 grid proxy exists on AFS for HTCondor to ship.

    On CERN lxplus, HTCondor submits via remote access points (bigbird schedds)
    which cannot read from the interactive node's local /tmp directory.
    If the proxy is located in /tmp, copy it to ~/private/x509up on AFS.
    """
    home = os.environ.get("HOME", "")
    afs_proxy = os.path.join(home, "private", "x509up")
    uid_proxy = f"/tmp/x509up_u{os.getuid()}"
    legacy_proxy = os.path.join(home, "x509up_file")

    source = proxy_arg
    if not source:
        if os.path.exists(uid_proxy) and os.path.exists(afs_proxy):
            source = uid_proxy if os.path.getmtime(uid_proxy) > os.path.getmtime(afs_proxy) else afs_proxy
        elif os.path.exists(afs_proxy):
            source = afs_proxy
        elif os.path.exists(legacy_proxy):
            source = legacy_proxy
        elif os.path.exists(uid_proxy):
            source = uid_proxy

    if not source or not os.path.exists(source):
        raise FileNotFoundError(
            f"No grid proxy found at '{source or afs_proxy}'. Run 'voms-proxy-init -voms cms' first."
        )

    if source.startswith("/tmp/"):
        os.makedirs(os.path.join(home, "private"), exist_ok=True)
        shutil.copy2(source, afs_proxy)
        os.chmod(afs_proxy, 0o600)
        print(f"Copied proxy from {source} to {afs_proxy} (AFS required by HTCondor)")
        return afs_proxy

    return source


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("produce", "score", "train"),
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
        help="Request GPUs; defaults to 1 for train stage",
    )
    parser.add_argument(
        "--job-flavour",
        default="workday",
        help="CERN HTCondor job flavour (e.g. espresso, microcentury, longlunch, workday, tomorrow)",
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help="Grid proxy to ship with the job (defaults to ~/private/x509up, auto-synced from /tmp)",
    )
    parser.add_argument(
        "--output-location",
        choices=("eos", "local"),
        default="eos",
        help="Destination for job outputs: 'eos' (default, writes to --eos-dir) or 'local' (Condor file transfer)",
    )
    parser.add_argument(
        "--eos-dir",
        default="/eos/user/n/nparmar/SVJScouting",
        help="EOS directory prefix when --output-location is 'eos'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="PyTorch model training random seed",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="List of random seeds to train multiple independent runs (train stage only)",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        default=False,
        help="Enable Comet ML tracking for training jobs",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        default=False,
        help="Explicitly disable Comet ML tracking for training jobs",
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


def queue_items(args, config):
    """One ``stage, sample, shard`` line per job."""
    if args.stage == "train":
        if args.seeds:
            return [f"{args.stage}, train_seed{seed}, 0" for seed in args.seeds]
        if args.seed is not None:
            return [f"{args.stage}, train_seed{args.seed}, 0"]
        return [f"{args.stage}, train, 0"]

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
    if args.stage == "train":
        pieces = [f"--config {args.config}"]
        if args.samples and "--train-on" not in args.job_arguments:
            pieces.append(f"--train-on {' '.join(args.samples)}")
        if args.output_location == "eos" and args.eos_dir and "--data-dir" not in args.job_arguments:
            pieces.append(f"--data-dir {args.eos_dir}")
        if args.comet and "--comet" not in args.job_arguments:
            pieces.append("--comet")
        if args.no_comet and "--no-comet" not in args.job_arguments:
            pieces.append("--no-comet")
        if args.seed is not None and not args.seeds:
            pieces.append(f"--seed {args.seed}")
    else:
        pieces = [f"--config {args.config}", f"--files-per-job {args.files_per_job}"]
    if args.job_arguments:
        pieces.append(args.job_arguments)
    return " ".join(pieces)


def main():
    args = parse_arguments()
    config = load_config(os.path.join(SVJ_GNN_DIR, args.config))

    proxy_path = resolve_and_prepare_proxy(args.proxy)
    if args.log_dir.startswith("/tmp"):
        print(f"WARNING: --log-dir '{args.log_dir}' is in local /tmp. On CERN lxplus, HTCondor access points and worker nodes cannot access local /tmp. Output and log files will fail to transfer. Use an AFS path instead.")

    os.makedirs(args.submit_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    tarball = os.path.join(args.submit_dir, "SVJ_GNN.tar.gz")
    if not args.skip_tarball:
        make_tarball(tarball)

    with open(os.path.join(CONDOR_DIR, "job.jdl.tmpl")) as template_file:
        template = template_file.read()

    num_gpus = args.request_gpus if args.request_gpus is not None else (1 if args.stage == "train" else 0)
    gpu_requirement = (
        f"request_gpus           = {num_gpus}"
        if num_gpus
        else "# no GPU requested"
    )

    env_vars = []
    if args.output_location == "eos":
        env_vars.append(f"GAE_OUTPUT_EOS={args.eos_dir}")
    api_key = os.environ.get("COMET_API_KEY", "8pkRGAjV7ZfkZfIKjdQQPCQuT")
    if api_key:
        env_vars.append(f"COMET_API_KEY={api_key}")

    env_setting = (
        f'Environment             = "{" ".join(env_vars)}"'
        if env_vars
        else "# No extra environment variables"
    )

    submit_file = template
    for placeholder, value in (
        ("CONDOR_DIR", CONDOR_DIR),
        ("TARBALL", tarball),
        ("X509_PROXY", proxy_path),
        ("LOG_DIR", args.log_dir),
        ("REQUEST_CPUS", str(args.request_cpus)),
        ("REQUEST_MEMORY", args.request_memory),
        ("REQUEST_DISK", args.request_disk),
        ("GPU_REQUIREMENT", gpu_requirement),
        ("JOB_FLAVOUR", args.job_flavour),
        ("ENVIRONMENT_SETTING", env_setting),
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
