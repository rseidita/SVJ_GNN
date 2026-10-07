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
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", default=None, help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--no-amp", action="store_true", default=None, help="Explicitly disable Automatic Mixed Precision")
    parser.add_argument(
        "--comet",
        action="store_true",
        default=False,
        help="Enable Comet ML tracking (requires comet-ml in venv)",
    )
    parser.add_argument("--no-comet", action="store_true", help="Explicitly disable Comet ML tracking")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def training_settings(config, args):
    settings = dict(config.training)
    for name in ("epochs", "batch_size", "learning_rate", "alpha", "seed"):
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


def start_experiment(config, settings, samples, args):
    """Return a comet experiment, or None when logging locally only."""
    want_comet = getattr(args, "comet", False) or (
        config.training.get("log_comet", False) and not args.no_comet
    )
    if args.no_comet or not want_comet:
        return None
    if comet_ml is None:
        print(
            "Notice: Comet ML was requested, but 'comet_ml' is not installed in the current environment.\n"
            "To enable Comet logging, create and source the layered venv:\n"
            "  source setup_gae_env.sh\n"
            "Continuing with local logging."
        )
        return None
    try:
        api_key = os.environ.get("COMET_API_KEY", "8pkRGAjV7ZfkZfIKjdQQPCQuT")
        experiment = comet_ml.Experiment(
            api_key=api_key,
            project_name=config.training.get("comet_project", "svj-gae"),
        )
    except Exception as error:
        print(f"Could not start a comet experiment ({error}); logging locally only.")
        return None

    experiment.log_parameters(settings)
    experiment.log_parameters(
        {f"graph_{key}": value for key, value in config.graph.items()}
    )
    experiment.log_parameter("model_spec", config.model.spec)
    experiment.log_parameters(
        {f"model_{key}": value for key, value in config.model.kwargs.items()}
    )
    experiment.log_parameter(
        "train_samples", ",".join(sample.name for sample in samples)
    )
    return experiment


def evaluate(model, loader, alpha, device, description, verbose, use_amp=False):
    """Mean loss over *loader*.

    Evaluated with the model in eval mode, so the edge term uses observed
    edges only, matching how inference scores are defined.
    """
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=description, disable=not verbose, leave=False):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                total, edges, features = training_loss(model, batch, alpha)
            totals += torch.tensor([total.item(), edges.item(), features.item()], dtype=torch.float64)
            n_batches += 1
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

    train_window = [list(settings["train_window"]) for _ in dataset_paths]
    validation_window = [
        list(settings["validation_window"]) for _ in dataset_paths
    ]
    test_window = [list(settings["test_window"]) for _ in dataset_paths]

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
        stats_loader = GraphDataLoader(
            dataset_paths=dataset_paths,
            split_windows=train_window,
            split_seed=settings["split_seed"],
            sampling="sequential",
            batch_size=settings["batch_size"],
        )
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
        means, stds = compute_normalization_stats(
            stats_loader, num_features, verbose=verbose
        )

    normalizer = OnTheFlyNormalizer(["x"], means, stds)

    loader_training = GraphDataLoader(
        dataset_paths=dataset_paths,
        split_windows=train_window,
        split_state=split_state,
        sampling=settings["sampling"],
        normalizer=normalizer,
        batch_size=settings["batch_size"],
    )
    loader_validation = GraphDataLoader(
        dataset_paths=dataset_paths,
        split_windows=validation_window,
        split_state=split_state,
        sampling="sequential",
        normalizer=normalizer,
        batch_size=settings["batch_size"],
    )
    loader_testing = GraphDataLoader(
        dataset_paths=dataset_paths,
        split_windows=test_window,
        split_state=split_state,
        sampling="sequential",
        normalizer=normalizer,
        batch_size=settings["batch_size"],
    )

    model = build_model(
        config.model.spec, num_features, **config.model.kwargs
    ).to(device)
    print(f"Built model {config.model.spec} ({num_features} node features) on {device}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"]
    )
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
        torch.save(payload, path)

    experiment = start_experiment(config, settings, samples, args)
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

            for batch_idx, batch in enumerate(
                tqdm(
                    loader_training,
                    desc=f"Epoch {epoch}",
                    disable=not verbose,
                    leave=False,
                )
            ):
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    total, edges, features = training_loss(model, batch, alpha)

                scaler.scale(total / args.accumulate_grad_batches).backward()
                if (batch_idx + 1) % args.accumulate_grad_batches == 0 or (
                    batch_idx + 1
                ) == len(loader_training):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                values = [total.item(), edges.item(), features.item()]
                history["batch_total"].append(values[0])
                history["batch_edges"].append(values[1])
                history["batch_nodes"].append(values[2])
                epoch_totals += torch.tensor(values, dtype=torch.float64)
                n_batches += 1

            if device.type == "cuda":
                torch.cuda.empty_cache()

            train_losses = (epoch_totals / max(n_batches, 1)).tolist()
            validation_losses = evaluate(
                model,
                loader_validation,
                alpha,
                device,
                f"Validation {epoch}",
                verbose,
                use_amp=use_amp,
            )
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
