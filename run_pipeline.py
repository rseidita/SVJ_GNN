#!/usr/bin/env python3
"""run_pipeline.py - Master orchestrator for the SVJ Graph Autoencoder pipeline.

Coordinates the end-to-end execution of the GAE pipeline or individual modular steps:
  1. Graph production: produce_graphs.py
  2. Autoencoder training: train_gae.py
  3. Streaming scoring: score_gae.py
  4. Discrimination plotting: plot_gae.py

Examples:
  # Run entire pipeline end-to-end:
  python run_pipeline.py --all --epochs 2 --device cpu --no-comet

  # Run specific steps:
  python run_pipeline.py --steps produce train --epochs 5
  python run_pipeline.py --steps score plot --checkpoint training/GraphAE_graphAE/checkpoint.pt

  # Score and plot in signal region:
  python run_pipeline.py --steps score plot --region signal_region
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.GaeConfig import load_config


VALID_STEPS = ("produce", "train", "score", "plot")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=VALID_STEPS,
        default=None,
        help="One or more pipeline steps to execute in order",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all pipeline steps: produce, train, score, plot",
    )
    parser.add_argument(
        "--config",
        default="gae_config.json",
        help="Path to GAE configuration file (default: gae_config.json)",
    )
    parser.add_argument(
        "--train-on",
        nargs="+",
        default=None,
        help="Sample name(s) to train on (default: data control region)",
    )
    parser.add_argument(
        "--produce-samples",
        nargs="+",
        default=None,
        help="Samples for produce_graphs.py (default: train-on sample)",
    )
    parser.add_argument(
        "--score-samples",
        nargs="+",
        default=None,
        help="Samples for score_gae.py (default: all configured samples)",
    )
    parser.add_argument(
        "--region",
        default="control",
        choices=["control", "signal_region", "signal", "validation", "highRT", "lowRT", "inclusive"],
        help="Selection region for scoring and discrimination (default: control)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs override",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training from an existing checkpoint file, or 'auto' to use the latest checkpoint in output-dir",
    )
    parser.add_argument(
        "--add-epochs",
        type=int,
        default=None,
        help="Number of additional epochs to train beyond checkpoint recorded epoch",
    )
    parser.add_argument(
        "--save-every-n-epochs",
        type=int,
        default=1,
        help="Frequency of saving intermediate checkpoints (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Gradient accumulation steps (useful for limited GPU memory)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Node feature loss weight alpha",
    )
    parser.add_argument(
        "--model-spec",
        default=None,
        help="Model architecture specification (e.g. GraphAE.graphAE)",
    )
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object string overriding model constructor kwargs",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run identifier for checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Execution device: auto, cuda, or cpu (default: auto)",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        default=False,
        help="Enable Comet ML tracking (optional; requires comet-ml in venv)",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        default=False,
        help="Explicitly disable Comet ML tracking",
    )
    parser.add_argument("--comet-experiment-key", default=None, help="Explicit Comet experiment key to resume")
    parser.add_argument("--new-comet-experiment", action="store_true", default=False, help="Force creating a new Comet experiment even when resuming training")
    parser.add_argument("--chunk-cache-size", "--gae-chunk-cache-size", dest="chunk_cache_size", type=int, default=None, help="In-memory chunk cache size override for training")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path to use for scoring and residual plots",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="History JSON path to use for loss curve plotting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for model initialization and training RNG (defaults to training.seed in config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for training output checkpoints (defaults to training.output_dir in config)",
    )
    parser.add_argument(
        "--scores-dir",
        default=None,
        help="Directory for score ROOT files (defaults to scoring.output_dir in config)",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory for output figures (defaults to plotting.plots_dir in config)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Max events to process per sample (useful for smoke runs)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for multiprocessing stages",
    )
    parser.add_argument("--pin-memory", action="store_true", default=None, help="Enable pinned host memory for DataLoader")
    parser.add_argument("--no-pin-memory", action="store_true", default=None, help="Explicitly disable pinned host memory")
    parser.add_argument("--negative-sampler", choices=["gpu", "cpu"], default="gpu", help="Negative edge sampler: 'gpu' (fast vectorized GPU sampler) or 'cpu' (legacy PyG sampler) (default: gpu)")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Batch size for validation and testing (default: 2 * train batch size)")
    parser.add_argument("--val-every-n-epochs", type=int, default=1, help="Frequency of running validation during training (default: 1)")
    parser.add_argument("--persistent-workers", action="store_true", default=None, help="Keep DataLoader worker processes alive across epochs")
    parser.add_argument("--no-persistent-workers", action="store_true", default=None, help="Explicitly disable persistent DataLoader worker processes")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress logs",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def run_stage(command, stage_name):
    print("\n" + "=" * 70)
    print(f"  PIPELINE STAGE: {stage_name.upper()}")
    print("=" * 70)
    print("Executing: " + " ".join(command))
    start_time = time.time()
    result = subprocess.run(command, cwd=SCRIPT_DIR)
    elapsed = time.time() - start_time
    if result.returncode != 0:
        print(f"\n[ERROR] Stage '{stage_name}' failed with return code {result.returncode} ({elapsed:.1f}s)")
        sys.exit(result.returncode)
    print(f"[SUCCESS] Stage '{stage_name}' finished in {elapsed:.1f}s")


def main():
    args = parse_arguments()
    config = load_config(os.path.join(SCRIPT_DIR, args.config))

    if not args.all and not args.steps:
        print("[ERROR] Please specify steps to run with --steps (produce train score plot) or use --all")
        sys.exit(1)

    steps_to_run = list(VALID_STEPS) if args.all else args.steps
    device = resolve_device(args.device)
    print(f"Pipeline initialized with steps: {steps_to_run}")
    print(f"Device: {device}")
    pers_str = "False" if args.no_persistent_workers else ("True" if args.persistent_workers else "auto (True when num_workers > 0)")
    pin_str = "False" if args.no_pin_memory else ("True" if args.pin_memory else "auto (True on CUDA)")
    val_bs_str = str(args.val_batch_size) if args.val_batch_size is not None else "auto (2x train batch size)"
    print(f"Pipeline options: negative_sampler={args.negative_sampler}, num_workers={args.num_workers}, val_batch_size={val_bs_str}, val_every_n_epochs={args.val_every_n_epochs}, persistent_workers={pers_str}, pin_memory={pin_str}, chunk_cache_size={args.chunk_cache_size}")

    # Resolve config-driven paths and active seed
    active_seed = args.seed if args.seed is not None else config.training.get("seed")
    training_output_dir = args.output_dir or config.training.get("output_dir", "training")
    # Determine run name and checkpoint path
    default_run_name = (args.model_spec or config.model.spec).replace(".", "_")
    if active_seed is not None:
        default_run_name = f"{default_run_name}_seed{active_seed}"
    run_name = args.run_name or default_run_name
    checkpoint_path = args.checkpoint or os.path.join(training_output_dir, run_name, "checkpoint.pt")
    history_path = args.history or os.path.join(training_output_dir, run_name, "history.json")

    base_scores = config.scoring.get("output_dir", "output_scores")
    scores_output_dir = args.scores_dir or (os.path.join(base_scores, run_name, f"seed{active_seed}") if active_seed is not None else os.path.join(base_scores, run_name))
    base_plots = config.plotting.get("plots_dir", "plots")
    plots_output_dir = args.plots_dir or (os.path.join(base_plots, run_name, f"seed{active_seed}") if active_seed is not None else os.path.join(base_plots, run_name))

    # Step 1: Produce Graphs
    if "produce" in steps_to_run:
        produce_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "produce_graphs.py"),
            "--config", args.config,
        ]
        produce_samples = args.produce_samples or args.train_on
        if produce_samples:
            produce_cmd.extend(["--samples", *produce_samples])
        if args.max_events:
            produce_cmd.extend(["--max-events", str(args.max_events)])
        if args.num_workers:
            produce_cmd.extend(["--num-workers", str(args.num_workers)])
        if args.quiet:
            produce_cmd.append("--quiet")
        run_stage(produce_cmd, "produce")

    # Step 2: Train Autoencoder
    if "train" in steps_to_run:
        train_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "train_gae.py"),
            "--config", args.config,
            "--output-dir", training_output_dir,
            "--device", device,
        ]
        if active_seed is not None:
            train_cmd.extend(["--seed", str(active_seed)])
        if args.train_on:
            train_cmd.extend(["--train-on", *args.train_on])
        if args.epochs:
            train_cmd.extend(["--epochs", str(args.epochs)])
        if args.batch_size:
            train_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.accumulate_grad_batches > 1:
            train_cmd.extend(["--accumulate-grad-batches", str(args.accumulate_grad_batches)])
        if args.learning_rate:
            train_cmd.extend(["--learning-rate", str(args.learning_rate)])
        if args.alpha:
            train_cmd.extend(["--alpha", str(args.alpha)])
        if args.model_spec:
            train_cmd.extend(["--model-spec", args.model_spec])
        if args.model_kwargs:
            train_cmd.extend(["--model-kwargs", args.model_kwargs])
        if args.run_name:
            train_cmd.extend(["--run-name", args.run_name])
        if args.resume is not None:
            if args.resume == "auto":
                train_cmd.append("--resume")
            else:
                train_cmd.extend(["--resume", args.resume])
        if args.add_epochs is not None:
            train_cmd.extend(["--add-epochs", str(args.add_epochs)])
        if args.save_every_n_epochs != 1:
            train_cmd.extend(["--save-every-n-epochs", str(args.save_every_n_epochs)])
        if args.comet:
            train_cmd.append("--comet")
        if args.no_comet:
            train_cmd.append("--no-comet")
        if args.comet_experiment_key:
            train_cmd.extend(["--comet-experiment-key", args.comet_experiment_key])
        if args.new_comet_experiment:
            train_cmd.append("--new-comet-experiment")
        if args.chunk_cache_size is not None:
            train_cmd.extend(["--chunk-cache-size", str(args.chunk_cache_size)])
        if args.num_workers is not None:
            train_cmd.extend(["--num-workers", str(args.num_workers)])
        if args.pin_memory:
            train_cmd.append("--pin-memory")
        if args.no_pin_memory:
            train_cmd.append("--no-pin-memory")
        if args.negative_sampler:
            train_cmd.extend(["--negative-sampler", args.negative_sampler])
        if args.val_batch_size is not None:
            train_cmd.extend(["--val-batch-size", str(args.val_batch_size)])
        if args.val_every_n_epochs != 1:
            train_cmd.extend(["--val-every-n-epochs", str(args.val_every_n_epochs)])
        if args.persistent_workers:
            train_cmd.append("--persistent-workers")
        if args.no_persistent_workers:
            train_cmd.append("--no-persistent-workers")
        if args.quiet:
            train_cmd.append("--quiet")
        run_stage(train_cmd, "train")

    # Step 3: Score ROOT Files
    if "score" in steps_to_run:
        if not os.path.exists(checkpoint_path):
            for candidate in ("checkpoint_best.pt", "checkpoint_latest.pt"):
                alt_path = os.path.join(training_output_dir, run_name, candidate)
                if os.path.exists(alt_path):
                    checkpoint_path = alt_path
                    break
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found at '{checkpoint_path}'. Run the 'train' step first or pass --checkpoint.")
            sys.exit(1)

        score_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "score_gae.py"),
            "--config", args.config,
            "--checkpoint", checkpoint_path,
            "--output-dir", scores_output_dir,
            "--region", args.region,
            "--device", device,
        ]
        if args.score_samples:
            score_cmd.extend(["--samples", *args.score_samples])
        if args.max_events:
            score_cmd.extend(["--max-events", str(args.max_events)])
        if args.num_workers:
            score_cmd.extend(["--num-workers", str(args.num_workers)])
        if args.quiet:
            score_cmd.append("--quiet")
        run_stage(score_cmd, "score")

    # Step 4: Plots & Discrimination Metrics
    if "plot" in steps_to_run:
        plot_cmd = [
            sys.executable,
            os.path.join(SCRIPT_DIR, "plot_gae.py"),
            "--config", args.config,
            "--scores-dir", scores_output_dir,
            "--outdir", plots_output_dir,
        ]
        if os.path.exists(history_path):
            plot_cmd.extend(["--history", history_path])
        if os.path.exists(checkpoint_path):
            plot_cmd.extend(["--checkpoint", checkpoint_path])
        if args.score_samples:
            plot_cmd.extend(["--samples", *args.score_samples])
        run_stage(plot_cmd, "plot")

    print("\n" + "=" * 70)
    print("  ALL REQUESTED PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()

