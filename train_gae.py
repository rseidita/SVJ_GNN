#!/usr/bin/env python3
"""Train a graph autoencoder on the graphs produced by produce_graphs.py.

The architecture is whatever gae_config.json names, so a new model in
architectures/ needs no change here.

    python train_gae.py --epochs 20
    python train_gae.py --train-on smoke --epochs 1 --no-comet

By default the autoencoder is trained on every sample with role "data"
(the control region). Background and signal are for inference and
discrimination plots only; use score_gae.py and plot_gae.py for those.
Use --train-on to restrict which data samples are used.
"""

# comet_ml must be imported before torch for its automatic logging to work.
try:
    import comet_ml
except ImportError:
    comet_ml = None

import argparse
import json
import os
import resource
import time

import torch
from tqdm import tqdm

from architectures.ModelRegistry import build_model, training_loss
from data_loading.graphDataLoader import GraphDataLoader
from utils.GaeConfig import load_config
from utils.LundTreeUtilities import (
    OnTheFlyNormalizer,
    compute_normalization_stats,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="gae_config.json")
    parser.add_argument(
        "--train-on",
        nargs="+",
        default=None,
        help="Data sample names to train on; defaults to every data sample",
    )
    parser.add_argument("--data-dir", default=None, help="Directory containing processed dataset directories (defaults to EOS or local output_dir)")
    parser.add_argument("--output-dir", default=None, help="Directory for training checkpoints; defaults to training.output_dir in config")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for model initialization and training RNG; defaults to training.seed in config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over (useful for limited GPU VRAM)",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--model-spec", default=None)
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object overriding the configured model kwargs",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Path to checkpoint file to resume from, or 'auto' to resume from latest checkpoint",
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
        help="Frequency of saving checkpoint_latest.pt during training (default: 1)",
    )
    parser.add_argument("--val-every-n-epochs", type=int, default=1, help="Frequency of running validation during training (default: 1)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=None, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-amp", action="store_true", default=None, help="Explicitly disable Automatic Mixed Precision")
    parser.add_argument("--comet", action="store_true", default=False, help="Enable Comet ML tracking (requires comet-ml in venv)")
    parser.add_argument("--no-comet", action="store_true", help="Explicitly disable Comet ML tracking")
    parser.add_argument("--comet-experiment-key", default=None, help="Explicit Comet experiment key to resume")
    parser.add_argument("--new-comet-experiment", action="store_true", default=False, help="Force creating a new Comet experiment even when resuming training")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--chunk-cache-size", "--gae-chunk-cache-size", dest="chunk_cache_size", type=int, default=None, help="In-memory chunk cache size override (overwrites GAE_CHUNK_CACHE_SIZE)")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of DataLoader worker processes (default: 0)")
    parser.add_argument("--pin-memory", action="store_true", default=None, help="Enable pinned host memory for DataLoader")
    parser.add_argument("--no-pin-memory", action="store_true", default=None, help="Explicitly disable pinned host memory")
    parser.add_argument("--negative-sampler", choices=["gpu", "cpu"], default="gpu", help="Negative edge sampler: 'gpu' (fast vectorized GPU sampler) or 'cpu' (legacy PyG sampler) (default: gpu)")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Batch size for validation and testing (default: 2 * train batch size)")
    parser.add_argument("--persistent-workers", action="store_true", default=None, help="Keep DataLoader worker processes alive across epochs")
    parser.add_argument("--no-persistent-workers", action="store_true", default=None, help="Explicitly disable persistent DataLoader worker processes")
    return parser.parse_args()


def training_settings(config, args):
    settings = dict(config.training)
    for name in ("epochs", "batch_size", "learning_rate", "alpha", "seed", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            settings[name] = value
    settings.setdefault("epochs", 1)
    settings.setdefault("batch_size", 64)
    settings.setdefault("learning_rate", 1e-3)
    settings.setdefault("alpha", 1.0)
    settings.setdefault("split_seed", 12345)
    settings.setdefault("sampling", "weighted")
    settings.setdefault("train_window", [0.0, 0.7])
    settings.setdefault("validation_window", [0.7, 0.85])
    settings.setdefault("test_window", [0.85, 1.0])
    return settings


def resolve_training_samples(config, args):
    if args.train_on:
        samples = config.select(args.train_on)
    else:
        samples = config.samples_with_role("data")
        if not samples:
            raise ValueError(
                "No sample has role 'data'; use --train-on to choose which "
                "data sample to train on"
            )
    for sample in samples:
        if sample.role not in ("data", "background"):
            raise ValueError(
                f"Cannot train on {sample.name!r} (role={sample.role!r}). "
                "The autoencoder is trained only on data and background in the control "
                "region. Score background and signal with score_gae.py for "
                "discrimination plots."
            )
    return samples


def start_experiment(config, settings, samples, args, resume_experiment_key=None, run_name=None):
    """Return a comet experiment, or None when logging locally only."""
    want_comet = getattr(args, "comet", False) or (config.training.get("log_comet", False) and not args.no_comet)
    if args.no_comet or not want_comet:
        return None
    if comet_ml is None:
        print("Notice: Comet ML was requested, but 'comet_ml' is not installed in the current environment.\nTo enable Comet logging, create and source the layered venv:\n  source setup_gae_env.sh\nContinuing with local logging.")
        return None

    base_project = config.training.get("comet_project", "svj-gae")
    project_name = f"{base_project}_{args.run_name}" if getattr(args, "run_name", None) else base_project

    api_key = os.environ.get("COMET_API_KEY", "8pkRGAjV7ZfkZfIKjdQQPCQuT")
    experiment = None
    if resume_experiment_key:
        try:
            print(f"Resuming existing Comet experiment: {resume_experiment_key} (project: {project_name})")
            experiment = comet_ml.ExistingExperiment(api_key=api_key, previous_experiment=resume_experiment_key, project_name=project_name)
        except Exception as error:
            print(f"Notice: Could not resume existing Comet experiment ({error}); starting a new experiment instead.")

    if experiment is None:
        try:
            experiment = comet_ml.Experiment(api_key=api_key, project_name=project_name)
        except Exception as error:
            print(f"Could not start a comet experiment ({error}); logging locally only.")
            return None

    if run_name and hasattr(experiment, "set_name"):
        experiment.set_name(run_name)

    experiment.log_parameters(settings)
    experiment.log_parameters({f"graph_{key}": value for key, value in config.graph.items()})
    experiment.log_parameter("model_spec", config.model.spec)
    experiment.log_parameters({f"model_{key}": value for key, value in config.model.kwargs.items()})
    experiment.log_parameter("train_samples", ",".join(sample.name for sample in samples))
    return experiment


def evaluate(model, loader, alpha, device, description, verbose, use_amp=False):
    """Mean loss over *loader*.

    Evaluated with the model in eval mode, so the edge term uses observed
    edges only, matching how inference scores are defined.
    """
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    n_batches = 0
    t_data_total = 0.0
    t_eval_total = 0.0
    t_prev = time.perf_counter()
    pbar = tqdm(loader, desc=description, disable=not verbose, leave=False)
    with torch.inference_mode():
        for batch_idx, batch in enumerate(pbar):
            t_data = time.perf_counter() - t_prev
            t_data_total += t_data
            t_eval_start = time.perf_counter()
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                total, edges, features = training_loss(model, batch, alpha)
            t_eval = time.perf_counter() - t_eval_start
            t_eval_total += t_eval
            totals += torch.tensor([total.item(), edges.item(), features.item()], dtype=torch.float64)
            n_batches += 1

            if batch_idx == 0 and verbose:
                try:
                    import psutil
                    proc = psutil.Process()
                    cpu_pct = proc.cpu_percent()
                    mem_rss = proc.memory_info().rss / (1024**2)
                except Exception:
                    cpu_pct = 0.0
                    mem_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                gpu_str = f"GPU allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB, reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB" if device.type == "cuda" else "Device: CPU"
                print(f"{description} Batch 0 diagnostics: Data I/O={t_data:.2f}s, Eval={t_eval:.2f}s | CPU RSS={mem_rss:.1f} MiB, CPU={cpu_pct:.1f}% | {gpu_str}")

            if verbose and (batch_idx + 1) % 25 == 0:
                avg_io = t_data_total / (batch_idx + 1)
                avg_eval = t_eval_total / (batch_idx + 1)
                pbar_str = f"I/O: {avg_io:.2f}s, Eval: {avg_eval:.2f}s"
                if device.type == "cuda":
                    pbar_str += f", GPU: {torch.cuda.memory_allocated(device) / 1024**2:.0f}/{torch.cuda.memory_reserved(device) / 1024**2:.0f}MB"
                pbar.set_postfix_str(pbar_str)

            t_prev = time.perf_counter()

    if verbose and n_batches > 0:
        t_loop_tot = t_data_total + t_eval_total
        pct_io = (t_data_total / t_loop_tot * 100) if t_loop_tot > 0 else 0
        pct_eval = (t_eval_total / t_loop_tot * 100) if t_loop_tot > 0 else 0
        try:
            import psutil
            proc = psutil.Process()
            cpu_pct = proc.cpu_percent()
            mem_rss = proc.memory_info().rss / (1024**2)
        except Exception:
            cpu_pct = 0.0
            mem_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        print(f"{description} timing diagnostics: Data I/O={t_data_total:.1f}s ({pct_io:.1f}%), Eval={t_eval_total:.1f}s ({pct_eval:.1f}%) | CPU RSS={mem_rss:.1f} MiB, CPU={cpu_pct:.1f}%")

    if not n_batches:
        return None
    return (totals / n_batches).tolist()


def main():
    args = parse_arguments()
    config = load_config(args.config)
    settings = training_settings(config, args)
    verbose = not args.quiet

    if args.model_spec is not None:
        config.model.spec = args.model_spec
    if args.model_kwargs is not None:
        config.model.kwargs = json.loads(args.model_kwargs)

    samples = resolve_training_samples(config, args)
    dataset_paths = []
    for sample in samples:
        dataset_paths.extend(config.all_dataset_paths(sample, base_dir=args.data_dir))
    print(
        "Training on "
        + ", ".join(f"{sample.name} ({sample.role})" for sample in samples)
    )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MiB")
        print(f"Cached/Reserved Memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MiB")
    else:
        print("CUDA is not available in this kernel.")
    use_amp = (device.type == "cuda") if args.amp is None and not args.no_amp else (bool(args.amp) and not args.no_amp)
    print(f"Automatic Mixed Precision (AMP): {'enabled' if use_amp else 'disabled'}")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))

    active_seed = settings.get("seed")
    if active_seed is not None:
        torch.manual_seed(active_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(active_seed)
        import numpy as np
        np.random.seed(active_seed)
        import random
        random.seed(active_seed)
        print(f"Using random seed: {active_seed}")

    base_output_dir = args.output_dir or config.training.get("output_dir", "training")
    default_run_name = config.model.spec.replace(".", "_")
    if active_seed is not None:
        default_run_name = f"{default_run_name}_seed{active_seed}"
    if args.run_name:
        if "{seed}" in args.run_name:
            run_name = args.run_name.format(seed=active_seed)
        elif active_seed is not None and not args.run_name.endswith(f"_seed{active_seed}"):
            run_name = f"{args.run_name}_seed{active_seed}"
        else:
            run_name = args.run_name
    else:
        run_name = default_run_name
    output_dir = os.path.join(base_output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Model: {config.model.spec}")
    print(f"Model checkpoint directory: {os.path.abspath(output_dir)}")
    if args.chunk_cache_size is not None:
        os.environ["GAE_CHUNK_CACHE_SIZE"] = str(args.chunk_cache_size)
    gae_cache_env = os.environ.get("GAE_CHUNK_CACHE_SIZE")
    effective_chunks = int(gae_cache_env) if gae_cache_env else 2
    print(f"Chunk cache configuration: GAE_CHUNK_CACHE_SIZE={gae_cache_env}, n_cached_chunks={effective_chunks}")

    checkpoint_data = None
    if args.resume is not None:
        if args.resume == "auto":
            candidates = [
                os.path.join(output_dir, "checkpoint_latest.pt"),
                os.path.join(output_dir, "checkpoint_best.pt"),
                os.path.join(output_dir, "checkpoint.pt"),
            ]
            resume_path = next((p for p in candidates if os.path.exists(p)), None)
            if resume_path is None:
                raise FileNotFoundError(
                    f"Auto-resume requested, but no checkpoint found in {output_dir}"
                )
        else:
            resume_path = args.resume
            if not os.path.exists(resume_path):
                raise FileNotFoundError(f"Checkpoint file '{resume_path}' does not exist.")

        print(f"Loading checkpoint for resumption from '{resume_path}'...")
        checkpoint_data = torch.load(resume_path, map_location="cpu", weights_only=False)

    resume_experiment_key = None
    if not args.new_comet_experiment:
        if args.comet_experiment_key:
            resume_experiment_key = args.comet_experiment_key
        elif checkpoint_data is not None:
            resume_experiment_key = checkpoint_data.get("comet_experiment_key")
            want_comet = getattr(args, "comet", False) or (config.training.get("log_comet", False) and not args.no_comet)
            if resume_experiment_key is None and want_comet:
                print("Notice: No Comet experiment key found in loaded checkpoint; creating a new Comet experiment.")

    train_window = [list(settings["train_window"]) for _ in dataset_paths]
    validation_window = [
        list(settings["validation_window"]) for _ in dataset_paths
    ]
    test_window = [list(settings["test_window"]) for _ in dataset_paths]

    num_workers = args.num_workers if args.num_workers is not None else settings.get("num_workers", 0)
    pin_memory = False if args.no_pin_memory else (True if args.pin_memory else settings.get("pin_memory", (device.type == "cuda")))
    val_batch_size = args.val_batch_size if args.val_batch_size is not None else (2 * settings["batch_size"])
    persistent_workers = False if (args.no_persistent_workers or num_workers == 0) else (True if args.persistent_workers else (num_workers > 0))

    if (
        checkpoint_data is not None
        and checkpoint_data.get("means") is not None
        and checkpoint_data.get("stds") is not None
    ):
        means = checkpoint_data["means"].to(device)
        stds = checkpoint_data["stds"].to(device)
        split_state = checkpoint_data.get("split_state")
        num_features = checkpoint_data.get(
            "num_features", len(means) if hasattr(means, "__len__") else 9
        )
        print("Reusing normalization statistics and split state from checkpoint.")
    else:
        # Normalization statistics come from the training partition only.
        stats_loader = GraphDataLoader(dataset_paths=dataset_paths, split_windows=train_window, split_seed=settings["split_seed"], sampling="sequential", batch_size=settings["batch_size"], num_workers=num_workers, pin_memory=pin_memory)
        print(f"Stats DataLoader: num_workers={stats_loader.num_workers}, pin_memory={stats_loader.pin_memory}")
        split_state = stats_loader.get_split_state()
        num_features = stats_loader.dataset[0].num_node_features
        if config.graph.get("use_lund_decomp"):
            n_lund = config.graph.get("n_lund_vars", 5)
            n_frac = 4 if config.graph.get("save_energy_fractions", True) else 0
            n_disp = 2 if config.graph.get("use_displacement", False) else 0
            n_4vec = 3 if config.graph.get("save_4vectors_in_lund_tree", False) else 0
            expected = n_lund + n_frac + n_disp + n_4vec
            if num_features != expected:
                raise ValueError(
                    f"Dataset feature mismatch: loaded dataset has {num_features} node features, "
                    f"but config.graph specifies {expected} features "
                    f"(save_4vectors_in_lund_tree={config.graph.get('save_4vectors_in_lund_tree', False)}). "
                    f"Training aborted to prevent training on an incompatible dataset."
                )
        means, stds = compute_normalization_stats(stats_loader, num_features, verbose=verbose)
        del stats_loader
        if torch.cuda.is_available():
            print(f"Device Name: {torch.cuda.get_device_name(0)}")
            print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MiB")
            print(f"Cached/Reserved Memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MiB")
        else:
            print("CUDA is not available in this kernel.")

    normalizer = OnTheFlyNormalizer(["x"], means, stds)

    loader_training = GraphDataLoader(dataset_paths=dataset_paths, split_windows=train_window, split_state=split_state, sampling=settings["sampling"], normalizer=normalizer, batch_size=settings["batch_size"], num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    print(f"Training DataLoader: num_workers={loader_training.num_workers}, pin_memory={loader_training.pin_memory}, persistent_workers={getattr(loader_training, 'persistent_workers', False)}")
    loader_validation = GraphDataLoader(dataset_paths=dataset_paths, split_windows=validation_window, split_state=split_state, sampling="sequential", normalizer=normalizer, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
    print(f"Validation DataLoader: num_workers={loader_validation.num_workers}, batch_size={val_batch_size}, pin_memory={loader_validation.pin_memory}, persistent_workers={getattr(loader_validation, 'persistent_workers', False)}")
    loader_testing = GraphDataLoader(dataset_paths=dataset_paths, split_windows=test_window, split_state=split_state, sampling="sequential", normalizer=normalizer, batch_size=val_batch_size, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)

    model = build_model(config.model.spec, num_features, **config.model.kwargs).to(device)
    if hasattr(model, "negative_sampler"):
        model.negative_sampler = args.negative_sampler
    print(f"Built model {config.model.spec} ({num_features} node features) on {device} (negative_sampler={getattr(model, 'negative_sampler', args.negative_sampler)})")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    alpha = settings["alpha"]

    start_epoch = 0
    best_val_loss = float("inf")
    history = {
        "batch_total": [],
        "batch_edges": [],
        "batch_nodes": [],
        "epoch_train": [],
        "epoch_validation": [],
    }

    if checkpoint_data is not None:
        if "state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["state_dict"])
            print("Loaded model state_dict.")
        if "optimizer_state_dict" in checkpoint_data and checkpoint_data["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            print("Loaded optimizer state_dict.")
        else:
            print("Notice: No optimizer_state_dict in checkpoint; initialized optimizer fresh.")
        if "scaler_state_dict" in checkpoint_data and checkpoint_data["scaler_state_dict"] is not None and use_amp and device.type == "cuda":
            scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
            print("Loaded scaler state_dict.")
        start_epoch = checkpoint_data.get("epoch", 0)
        best_val_loss = checkpoint_data.get("best_val_loss", float("inf"))
        if "history" in checkpoint_data and isinstance(checkpoint_data["history"], dict):
            history = checkpoint_data["history"]
            print(f"Restored training history with {len(history.get('epoch_train', []))} recorded epochs.")

    if args.add_epochs is not None:
        total_epochs = start_epoch + args.add_epochs
    else:
        total_epochs = settings["epochs"]

    print(f"Training schedule: starting at epoch {start_epoch}, targeting total {total_epochs} epochs.")
    experiment = None

    def save_checkpoint(path, current_epoch, current_best_loss):
        saved_graph = dict(config.graph)
        if saved_graph.get("use_lund_decomp"):
            n_lund = saved_graph.get("n_lund_vars", 5)
            n_frac = 4 if saved_graph.get("save_energy_fractions", True) else 0
            n_disp = 2 if saved_graph.get("use_displacement", False) else 0
            features_without_4vec = n_lund + n_frac + n_disp
            features_with_4vec = features_without_4vec + 3
            if num_features == features_without_4vec:
                saved_graph["save_4vectors_in_lund_tree"] = False
            elif num_features == features_with_4vec:
                saved_graph["save_4vectors_in_lund_tree"] = True
        payload = {
            "epoch": current_epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if (use_amp and device.type == "cuda") else None,
            "history": history,
            "best_val_loss": current_best_loss,
            "model_spec": config.model.spec,
            "model_kwargs": config.model.kwargs,
            "num_features": num_features,
            "means": means,
            "stds": stds,
            "split_state": split_state,
            "alpha": alpha,
            "seed": active_seed,
            "graph": saved_graph,
            "training": settings,
            "train_samples": [sample.name for sample in samples],
        }
        comet_key = None
        if experiment is not None:
            comet_key = experiment.get_key() if hasattr(experiment, "get_key") else getattr(experiment, "id", None)
        elif checkpoint_data is not None:
            comet_key = checkpoint_data.get("comet_experiment_key")
        payload["comet_experiment_key"] = comet_key
        torch.save(payload, path)

    experiment = start_experiment(config, settings, samples, args, resume_experiment_key=resume_experiment_key, run_name=run_name)
    if experiment is not None and checkpoint_data is not None:
        experiment.log_parameter("resumed_from_epoch", start_epoch)

    if start_epoch >= total_epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch}, which meets or exceeds "
            f"target total epochs {total_epochs}. Skipping training loop."
        )
    else:
        start = time.time()
        for epoch in range(start_epoch, total_epochs):
            model.train()
            epoch_totals = torch.zeros(3, dtype=torch.float64)
            n_batches = 0
            optimizer.zero_grad()

            t_data_total = 0.0
            t_loss_total = 0.0
            t_bwd_total = 0.0
            t_prev = time.perf_counter()
            pbar = tqdm(loader_training, desc=f"Epoch {epoch}", disable=not verbose, leave=False)

            for batch_idx, batch in enumerate(pbar):
                t_data = time.perf_counter() - t_prev
                t_data_total += t_data

                t_loss_start = time.perf_counter()
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    total, edges, features = training_loss(model, batch, alpha)
                t_loss = time.perf_counter() - t_loss_start
                t_loss_total += t_loss

                t_bwd_start = time.perf_counter()
                scaler.scale(total / args.accumulate_grad_batches).backward()
                if (batch_idx + 1) % args.accumulate_grad_batches == 0 or (
                    batch_idx + 1
                ) == len(loader_training):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                t_bwd = time.perf_counter() - t_bwd_start
                t_bwd_total += t_bwd

                values = [total.item(), edges.item(), features.item()]
                history["batch_total"].append(values[0])
                history["batch_edges"].append(values[1])
                history["batch_nodes"].append(values[2])
                epoch_totals += torch.tensor(values, dtype=torch.float64)
                n_batches += 1

                if batch_idx == 0 and verbose:
                    try:
                        import psutil
                        proc = psutil.Process()
                        cpu_pct = proc.cpu_percent()
                        mem_rss = proc.memory_info().rss / (1024**2)
                    except Exception:
                        cpu_pct = 0.0
                        mem_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    gpu_str = f"GPU allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MiB, reserved: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MiB" if device.type == "cuda" else "Device: CPU"
                    print(f"Epoch {epoch} Batch 0 diagnostics: Data I/O={t_data:.2f}s, Loss={t_loss:.2f}s, Bwd+Opt={t_bwd:.2f}s | CPU RSS={mem_rss:.1f} MiB, CPU={cpu_pct:.1f}% | {gpu_str}")

                if verbose and (batch_idx + 1) % 25 == 0:
                    avg_io = t_data_total / (batch_idx + 1)
                    avg_loss = t_loss_total / (batch_idx + 1)
                    avg_bwd = t_bwd_total / (batch_idx + 1)
                    pbar_str = f"I/O: {avg_io:.2f}s, Loss: {avg_loss:.2f}s, Bwd: {avg_bwd:.2f}s"
                    if device.type == "cuda":
                        pbar_str += f", GPU: {torch.cuda.memory_allocated(device) / 1024**2:.0f}/{torch.cuda.memory_reserved(device) / 1024**2:.0f}MB"
                    pbar.set_postfix_str(pbar_str)

                t_prev = time.perf_counter()

            if verbose and n_batches > 0:
                t_loop_tot = t_data_total + t_loss_total + t_bwd_total
                pct_io = (t_data_total / t_loop_tot * 100) if t_loop_tot > 0 else 0
                pct_loss = (t_loss_total / t_loop_tot * 100) if t_loop_tot > 0 else 0
                pct_bwd = (t_bwd_total / t_loop_tot * 100) if t_loop_tot > 0 else 0
                try:
                    import psutil
                    proc = psutil.Process()
                    cpu_pct = proc.cpu_percent()
                    mem_rss = proc.memory_info().rss / (1024**2)
                except Exception:
                    cpu_pct = 0.0
                    mem_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                print(f"Epoch {epoch} timing diagnostics: Data I/O={t_data_total:.1f}s ({pct_io:.1f}%), Loss={t_loss_total:.1f}s ({pct_loss:.1f}%), Bwd+Opt={t_bwd_total:.1f}s ({pct_bwd:.1f}%) | CPU RSS={mem_rss:.1f} MiB, CPU={cpu_pct:.1f}%")

            if device.type == "cuda":
                torch.cuda.empty_cache()

            train_losses = (epoch_totals / max(n_batches, 1)).tolist()
            should_validate = ((epoch + 1 - start_epoch) % args.val_every_n_epochs == 0) or (epoch + 1 == total_epochs)
            if should_validate:
                validation_losses = evaluate(model, loader_validation, alpha, device, f"Validation {epoch}", verbose, use_amp=use_amp)
            else:
                validation_losses = None
            history["epoch_train"].append(train_losses)
            history["epoch_validation"].append(validation_losses)

            is_best = False
            if validation_losses is not None:
                val_total = validation_losses[0]
                if val_total < best_val_loss:
                    best_val_loss = val_total
                    is_best = True

            message = (
                f"Epoch {epoch}: train total {train_losses[0]:.5f} "
                f"(edges {train_losses[1]:.5f}, nodes {train_losses[2]:.5f})"
            )
            if validation_losses is not None:
                message += (
                    f", val total {validation_losses[0]:.5f} "
                    f"(edges {validation_losses[1]:.5f}, nodes {validation_losses[2]:.5f})"
                )
            print(message)
            if torch.cuda.is_available():
                print(f"Device Name: {torch.cuda.get_device_name(0)}")
                print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MiB")
                print(f"Cached/Reserved Memory: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MiB")
            else:
                print("CUDA is not available in this kernel.")

            if experiment is not None:
                experiment.log_metric("train_total", train_losses[0], epoch=epoch, step=epoch)
                experiment.log_metric("train_edges", train_losses[1], epoch=epoch, step=epoch)
                experiment.log_metric("train_nodes", train_losses[2], epoch=epoch, step=epoch)
                if validation_losses is not None:
                    experiment.log_metric("validation_total", validation_losses[0], epoch=epoch, step=epoch)
                    experiment.log_metric("validation_edges", validation_losses[1], epoch=epoch, step=epoch)
                    experiment.log_metric("validation_nodes", validation_losses[2], epoch=epoch, step=epoch)

            if (epoch + 1 - start_epoch) % args.save_every_n_epochs == 0 or (epoch + 1 == total_epochs):
                save_checkpoint(
                    os.path.join(output_dir, "checkpoint_latest.pt"),
                    epoch + 1,
                    best_val_loss,
                )
                if is_best:
                    save_checkpoint(
                        os.path.join(output_dir, "checkpoint_best.pt"),
                        epoch + 1,
                        best_val_loss,
                    )

        print(f"--> Training completed in {time.time() - start:.2f} seconds")

    test_losses = evaluate(
        model, loader_testing, alpha, device, "Test", verbose, use_amp=use_amp
    )
    if test_losses is not None:
        history["test"] = test_losses
        print(f"Held-out test total loss: {test_losses[0]:.5f}")
        if experiment is not None:
            experiment.log_metric("test_total", test_losses[0])

    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    save_checkpoint(checkpoint_path, total_epochs, best_val_loss)

    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as history_file:
        json.dump(history, history_file, indent=2)

    print(f"Model: {config.model.spec}")
    print(f"Saved model checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Saved training history: {os.path.abspath(history_path)}")

    if experiment is not None:
        experiment.log_asset(checkpoint_path)
        experiment.log_asset(history_path)
        experiment.end()


if __name__ == "__main__":
    main()
