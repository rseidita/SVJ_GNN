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
    "SVJ_GNN/data",
    "SVJ_GNN/data_aimar",
    "SVJ_GNN/scores",
    "SVJ_GNN/output_scores",
    "SVJ_GNN/output_scores_aimar",
    "SVJ_GNN/data_aimar_graphs_V3"
    "SVJ_GNN/plots",
    "SVJ_GNN/plots_aimar",
    "SVJ_GNN/training",
    "SVJ_GNN/training_aimar",
    "SVJ_GNN/condor/logs",
    "*.ipynb",
    "*.root",
    "__pycache__",
    "SVJ_GNN/condor/submit/*",
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
    if source and os.path.isdir(source):
        source = os.path.join(source, f"x509up_u{os.getuid()}")
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
    parser.add_argument("--log-dir", default=os.path.join(CONDOR_DIR, "logs"), help="Directory for job stdout/stderr/log (default: condor/logs)")
    parser.add_argument("--request-cpus", type=int, default=1)
    parser.add_argument("--request-memory", default="8 GB")
    parser.add_argument("--request-disk", default="4 GB")
    parser.add_argument(
        "--request-gpus",
        type=int,
        default=None,
        help="Request GPUs; defaults to 1 for train stage",
    )
    parser.add_argument(
        "--min-gpu-memory",
        type=int,
        default=10000,
        help="Minimum GPU VRAM in MB (default: 10000 MB to target 10.0 GB+ V100/A100/H100)",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=None, help="Total epochs for training")
    parser.add_argument("--amp", action="store_true", default=None, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-amp", action="store_true", default=None, help="Disable Automatic Mixed Precision (AMP)")
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
        help="List of random seeds to train or score multiple independent runs",
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
        "--checkpoint",
        default=None,
        help="Checkpoint path for scoring stage (defaults to previous train stage output location)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Custom run name for training output checkpoints and logs (train stage only)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training: 'auto' (default if flag given without value) or path to checkpoint file (train stage only)",
    )
    parser.add_argument(
        "--add-epochs",
        type=int,
        default=None,
        help="Number of additional epochs to train beyond checkpoint recorded epoch (train stage only)",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=None,
        help="Frequency of saving checkpoint_latest.pt during training (train stage only)",
    )
    parser.add_argument("--accumulate-grad-batches", type=int, default=None, help="Number of batches to accumulate gradients over (useful for limited GPU memory)")
    parser.add_argument("--chunk-cache-size", "--gae-chunk-cache-size", dest="chunk_cache_size", type=int, default=None, help="In-memory chunk cache size override for training")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of worker processes for multiprocessing stages")
    parser.add_argument("--pin-memory", action="store_true", default=None, help="Enable pinned host memory for DataLoader")
    parser.add_argument("--no-pin-memory", action="store_true", default=None, help="Explicitly disable pinned host memory")
    parser.add_argument("--negative-sampler", choices=["gpu", "cpu"], default="gpu", help="Negative edge sampler: 'gpu' (fast vectorized GPU sampler) or 'cpu' (legacy PyG sampler) (default: gpu)")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Batch size for validation and testing (default: 2 * train batch size)")
    parser.add_argument("--val-every-n-epochs", type=int, default=1, help="Frequency of running validation during training (default: 1)")
    parser.add_argument("--persistent-workers", action="store_true", default=None, help="Keep DataLoader worker processes alive across epochs")
    parser.add_argument("--no-persistent-workers", action="store_true", default=None, help="Explicitly disable persistent DataLoader worker processes")
    parser.add_argument(
        "--job-arguments",
        default="",
        help="Extra arguments appended to the stage's python command",
    )
    parser.add_argument("--skip-tarball", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Generate submit JDL and tarball without calling condor_submit")
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
    """One ``stage, sample, shard, log_subdir, extra_args`` line per job."""
    if args.stage == "train":
        sample_tag = args.run_name or ("_".join(args.samples) if args.samples else "train")
        if args.seeds:
            return [
                f"{args.stage}, {sample_tag}_seed{seed}, 0, "
                + (f"train/{args.run_name}/seed{seed}" if args.run_name else f"train/seed{seed}")
                + f", --seed {seed}"
                for seed in args.seeds
            ]
        if args.seed is not None:
            log_sub = f"train/{args.run_name}/seed{args.seed}" if args.run_name else f"train/seed{args.seed}"
            return [f"{args.stage}, {sample_tag}_seed{args.seed}, 0, {log_sub}, --seed {args.seed}"]
        log_sub = f"train/{args.run_name}" if args.run_name else "train"
        return [f"{args.stage}, {sample_tag}, 0, {log_sub}, "]

    if args.stage == "score":
        base_scoring_dir = config.scoring.get("output_dir", "output_scores")
        seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [None])
        items = []
        for seed in seeds:
            ckpt_path = config.find_checkpoint(
                args.checkpoint,
                run_name=args.run_name,
                seed=seed,
                base_dir=args.eos_dir if args.output_location == "eos" else None,
            )
            seed_label = f" (seed {seed})" if seed is not None else ""
            print(f"Model checkpoint path{seed_label}: {ckpt_path}")
            if args.run_name:
                score_out_rel = os.path.join(base_scoring_dir, args.run_name, f"seed{seed}") if seed is not None else os.path.join(base_scoring_dir, args.run_name)
                log_sub = os.path.join("score", args.run_name, f"seed{seed}") if seed is not None else os.path.join("score", args.run_name)
            else:
                score_out_rel = os.path.join(base_scoring_dir, f"seed{seed}") if seed is not None else base_scoring_dir
                log_sub = os.path.join("score", f"seed{seed}") if seed is not None else "score"

            extra_parts = [f"--checkpoint {ckpt_path}", f"--output-dir {score_out_rel}"]
            if seed is not None:
                extra_parts.append(f"--seed {seed}")
            extra_args = " ".join(extra_parts)

            for sample in config.select(args.samples):
                n_shards = len(shard_files(sample.files(), args.files_per_job))
                if seed == seeds[0]:
                    print(f"Sample '{sample.name}': {n_shards} shard(s)")
                items.extend(
                    f"{args.stage}, {sample.name}, {shard}, {log_sub}, {extra_args}"
                    for shard in range(n_shards)
                )
        return items

    items = []
    for sample in config.select(args.samples):
        n_shards = len(shard_files(sample.files(), args.files_per_job))
        print(f"Sample '{sample.name}': {n_shards} shard(s)")
        items.extend(
            f"{args.stage}, {sample.name}, {shard}, produce, " for shard in range(n_shards)
        )
    return items


def job_arguments(args, config):
    """Arguments appended after the stage, sample and shard."""
    if args.stage == "train":
        print(f"Model: {config.model.spec}")
        pieces = [f"--config {args.config}"]
        if args.samples and "--train-on" not in args.job_arguments:
            pieces.append(f"--train-on {' '.join(args.samples)}")
        if args.output_location == "eos" and args.eos_dir and "--data-dir" not in args.job_arguments:
            pieces.append(f"--data-dir {args.eos_dir}")
        if args.run_name and "--run-name" not in args.job_arguments:
            pieces.append(f"--run-name {args.run_name}")
        if args.batch_size is not None and "--batch-size" not in args.job_arguments:
            pieces.append(f"--batch-size {args.batch_size}")
        if args.learning_rate is not None and "--learning-rate" not in args.job_arguments:
            pieces.append(f"--learning-rate {args.learning_rate}")
        if args.epochs is not None and "--epochs" not in args.job_arguments:
            pieces.append(f"--epochs {args.epochs}")
        if args.amp and "--amp" not in args.job_arguments:
            pieces.append("--amp")
        if args.no_amp and "--no-amp" not in args.job_arguments:
            pieces.append("--no-amp")
        if args.comet and "--comet" not in args.job_arguments:
            pieces.append("--comet")
        if args.no_comet and "--no-comet" not in args.job_arguments:
            pieces.append("--no-comet")
        if args.resume is not None and "--resume" not in args.job_arguments:
            pieces.append(f"--resume {args.resume}" if args.resume != "auto" else "--resume")
            print(f"Training resume mode: {args.resume}")
        if args.add_epochs is not None and "--add-epochs" not in args.job_arguments:
            pieces.append(f"--add-epochs {args.add_epochs}")
        if args.save_every_n_epochs is not None and "--save-every-n-epochs" not in args.job_arguments:
            pieces.append(f"--save-every-n-epochs {args.save_every_n_epochs}")
        if args.accumulate_grad_batches is not None and "--accumulate-grad-batches" not in args.job_arguments:
            pieces.append(f"--accumulate-grad-batches {args.accumulate_grad_batches}")
        if args.chunk_cache_size is not None and "--chunk-cache-size" not in args.job_arguments:
            pieces.append(f"--chunk-cache-size {args.chunk_cache_size}")
        if args.num_workers is not None and "--num-workers" not in args.job_arguments:
            pieces.append(f"--num-workers {args.num_workers}")
        if args.pin_memory and "--pin-memory" not in args.job_arguments:
            pieces.append("--pin-memory")
        if args.no_pin_memory and "--no-pin-memory" not in args.job_arguments:
            pieces.append("--no-pin-memory")
        if args.negative_sampler and "--negative-sampler" not in args.job_arguments:
            pieces.append(f"--negative-sampler {args.negative_sampler}")
        if args.val_batch_size is not None and "--val-batch-size" not in args.job_arguments:
            pieces.append(f"--val-batch-size {args.val_batch_size}")
        if args.val_every_n_epochs != 1 and "--val-every-n-epochs" not in args.job_arguments:
            pieces.append(f"--val-every-n-epochs {args.val_every_n_epochs}")
        if args.persistent_workers and "--persistent-workers" not in args.job_arguments:
            pieces.append("--persistent-workers")
        if args.no_persistent_workers and "--no-persistent-workers" not in args.job_arguments:
            pieces.append("--no-persistent-workers")
    elif args.stage == "score":
        pieces = [f"--config {args.config}", f"--files-per-job {args.files_per_job}"]
        if args.run_name and "--run-name" not in args.job_arguments:
            pieces.append(f"--run-name {args.run_name}")
    else:
        pieces = [f"--config {args.config}", f"--files-per-job {args.files_per_job}"]
    if args.job_arguments:
        pieces.append(args.job_arguments)
    return " ".join(pieces)


def prepare_output_directories(args, config):
    """Pre-create destination directories before submitting batch jobs to prevent worker race conditions."""
    base_dir = args.eos_dir if args.output_location == "eos" else "."

    if args.stage == "produce":
        prod_dir = config.production.get("output_dir", "data")
        samples = config.select(args.samples)
        for sample in samples:
            target_processed = os.path.join(base_dir, prod_dir, sample.name, "processed")
            os.makedirs(target_processed, exist_ok=True)
            print(f"Pre-created directory: {target_processed}")
        target_log_dir = os.path.join(args.log_dir, "produce")
        os.makedirs(target_log_dir, exist_ok=True)
        print(f"Pre-created log directory: {target_log_dir}")
    elif args.stage == "score":
        base_scoring_dir = config.scoring.get("output_dir", "output_scores")
        seeds = args.seeds if args.seeds else ([args.seed] if args.seed is not None else [None])
        for seed in seeds:
            if args.run_name:
                score_out_rel = os.path.join(base_scoring_dir, args.run_name, f"seed{seed}") if seed is not None else os.path.join(base_scoring_dir, args.run_name)
                log_sub = os.path.join("score", args.run_name, f"seed{seed}") if seed is not None else os.path.join("score", args.run_name)
            else:
                score_out_rel = os.path.join(base_scoring_dir, f"seed{seed}") if seed is not None else base_scoring_dir
                log_sub = os.path.join("score", f"seed{seed}") if seed is not None else "score"
            target_dir = os.path.join(base_dir, score_out_rel)
            os.makedirs(target_dir, exist_ok=True)
            print(f"Pre-created directory: {target_dir}")
            target_log_dir = os.path.join(args.log_dir, log_sub)
            os.makedirs(target_log_dir, exist_ok=True)
            print(f"Pre-created log directory: {target_log_dir}")
    elif args.stage == "train":
        train_dir = config.training.get("output_dir", "training")
        run_name = args.run_name or config.model.spec.replace(".", "_")
        if args.seeds:
            for seed in args.seeds:
                target_dir = os.path.join(base_dir, train_dir, f"{run_name}_seed{seed}")
                os.makedirs(target_dir, exist_ok=True)
                print(f"Pre-created directory: {target_dir}")
                target_log_dir = os.path.join(args.log_dir, "train", run_name, f"seed{seed}")
                os.makedirs(target_log_dir, exist_ok=True)
                print(f"Pre-created log directory: {target_log_dir}")
        elif args.seed is not None:
            target_dir = os.path.join(base_dir, train_dir, f"{run_name}_seed{args.seed}")
            os.makedirs(target_dir, exist_ok=True)
            print(f"Pre-created directory: {target_dir}")
            target_log_dir = os.path.join(args.log_dir, "train", run_name, f"seed{args.seed}")
            os.makedirs(target_log_dir, exist_ok=True)
            print(f"Pre-created log directory: {target_log_dir}")
        else:
            target_dir = os.path.join(base_dir, train_dir, run_name)
            os.makedirs(target_dir, exist_ok=True)
            print(f"Pre-created directory: {target_dir}")
            target_log_dir = os.path.join(args.log_dir, "train", run_name)
            os.makedirs(target_log_dir, exist_ok=True)
            print(f"Pre-created log directory: {target_log_dir}")


def main():
    args = parse_arguments()
    config = load_config(os.path.join(SVJ_GNN_DIR, args.config))

    req_cpus = args.request_cpus if args.request_cpus is not None else 1
    if args.stage == "train" and args.num_workers is not None and args.request_cpus == 1 and args.num_workers > 1:
        req_cpus = args.num_workers

    if args.stage == "train":
        pers_str = "False" if args.no_persistent_workers else ("True" if args.persistent_workers else "auto (True when num_workers > 0)")
        pin_str = "False" if args.no_pin_memory else ("True" if args.pin_memory else "auto (True on CUDA)")
        val_bs_str = str(args.val_batch_size) if args.val_batch_size is not None else "auto (2x train batch size)"
        print(f"Pipeline options: negative_sampler={args.negative_sampler}, num_workers={args.num_workers}, val_batch_size={val_bs_str}, val_every_n_epochs={args.val_every_n_epochs}, persistent_workers={pers_str}, pin_memory={pin_str}, chunk_cache_size={args.chunk_cache_size}")

    proxy_path = resolve_and_prepare_proxy(args.proxy)
    if args.log_dir.startswith("/tmp"):
        print(f"WARNING: --log-dir '{args.log_dir}' is in local /tmp. On CERN lxplus, HTCondor access points and worker nodes cannot access local /tmp across machines. Output and log files will fail to transfer. Use an AFS path instead.")

    os.makedirs(args.submit_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    prepare_output_directories(args, config)

    tarball = os.path.join(args.submit_dir, "SVJ_GNN.tar.gz")
    if not args.skip_tarball:
        make_tarball(tarball)

    with open(os.path.join(CONDOR_DIR, "job.jdl.tmpl")) as template_file:
        template = template_file.read()

    num_gpus = args.request_gpus if args.request_gpus is not None else (1 if args.stage == "train" else 0)
    min_gpu_mem = args.min_gpu_memory
    gpu_requirement = (
        f"request_gpus           = {num_gpus}\n"
        # f"requirements            = (TARGET.GPUs_Capability >= 7.0)"
        # + (f" && (TARGET.GPUs_GlobalMemoryMb >= {min_gpu_mem})" if min_gpu_mem else "")
        + f"\nRank                    = TARGET.GPUs_Capability * 1000 + TARGET.GPUs_GlobalMemoryMb"
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
    queued_jobs = queue_items(args, config)
    for placeholder, value in (
        ("CONDOR_DIR", CONDOR_DIR),
        ("TARBALL", tarball),
        ("X509_PROXY", proxy_path),
        ("LOG_DIR", args.log_dir),
        ("REQUEST_CPUS", str(req_cpus)),
        ("REQUEST_MEMORY", args.request_memory),
        ("REQUEST_DISK", args.request_disk),
        ("GPU_REQUIREMENT", gpu_requirement),
        ("JOB_FLAVOUR", args.job_flavour),
        ("ENVIRONMENT_SETTING", env_setting),
        ("JOB_ARGUMENTS", job_arguments(args, config)),
        ("QUEUE_ITEMS", "\n".join(queued_jobs)),
    ):
        submit_file = submit_file.replace(placeholder, value)

    submit_path = os.path.join(args.submit_dir, f"{args.stage}.jdl")
    with open(submit_path, "w") as output_file:
        output_file.write(submit_file)

    print(f"Wrote {submit_path}")
    print(f"Submit it with: condor_submit {submit_path}")
    if not args.dry_run:
        try:

            subprocess.run(["condor_submit", submit_path], check=True)
        except OSError as error:
            if error.errno !=8:
                raise
            subprocess.run(["bash", shutil.which("condor_submit"), submit_path], check=True)    
    else:
        print("[dry-run] JDL generated successfully. Condor submission skipped.")

    log_dirs = sorted({
        os.path.join(args.log_dir, item.split(", ", 4)[3])
        for item in queued_jobs
    })
    print("\nLog file locations:")
    for d in log_dirs:
        print(f"  Directory: {d}")
        print(f"    - stdout: {d}/<sample>_<shard>.stdout")
        print(f"    - stderr: {d}/<sample>_<shard>.stderr")
        print(f"    - log:    {d}/<sample>_<shard>.log")


if __name__ == "__main__":
    main()
