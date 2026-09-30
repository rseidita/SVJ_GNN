#!/usr/bin/env python3
"""Train a graph autoencoder on the graphs produced by produce_graphs.py.

The architecture is whatever gae_config.json names, so a new model in
architectures/ needs no change here.

    python train_gae.py --epochs 20
    python train_gae.py --train-on smoke --epochs 1 --no-comet

By default the autoencoder is trained on every sample with role
"background". Use --train-on to train on named samples instead.
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
        help="Sample names to train on; defaults to every background sample",
    )
    parser.add_argument("--output-dir", default="training")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--model-spec", default=None)
    parser.add_argument(
        "--model-kwargs",
        default=None,
        help="JSON object overriding the configured model kwargs",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-comet", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def training_settings(config, args):
    settings = dict(config.training)
    for name in ("epochs", "batch_size", "learning_rate", "alpha"):
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
        samples = [config.sample(name) for name in args.train_on]
    else:
        samples = config.samples_with_role("background")
        if not samples:
            raise ValueError(
                "No sample has role 'background'; use --train-on to choose "
                "which samples to train on"
            )
    return samples


def start_experiment(config, settings, samples, args):
    """Return a comet experiment, or None when logging locally only."""
    if args.no_comet:
        return None
    if comet_ml is None:
        print(
            "comet_ml is not installed; logging locally only. Install it with "
            "'python -m pip install -r requirements-gae.txt'."
        )
        return None
    try:
        experiment = comet_ml.Experiment(
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


def evaluate(model, loader, alpha, device, description, verbose):
    """Mean loss over *loader*.

    Evaluated with the model in eval mode, so the edge term uses observed
    edges only, matching how inference scores are defined.
    """
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64)
    n_batches = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=description, disable=not verbose, leave=False):
            batch = batch.to(device)
            total, edges, features = training_loss(model, batch, alpha)
            totals += torch.tensor(
                [total.item(), edges.item(), features.item()],
                dtype=torch.float64,
            )
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
    dataset_paths = [config.dataset_paths(sample) for sample in samples]
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

    train_window = [list(settings["train_window"]) for _ in dataset_paths]
    validation_window = [
        list(settings["validation_window"]) for _ in dataset_paths
    ]
    test_window = [list(settings["test_window"]) for _ in dataset_paths]

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
    optimizer = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"]
    )
    alpha = settings["alpha"]

    experiment = start_experiment(config, settings, samples, args)
    history = {
        "batch_total": [],
        "batch_edges": [],
        "batch_nodes": [],
        "epoch_train": [],
        "epoch_validation": [],
    }

    start = time.time()
    for epoch in range(settings["epochs"]):
        model.train()
        epoch_totals = torch.zeros(3, dtype=torch.float64)
        n_batches = 0

        for batch in tqdm(
            loader_training,
            desc=f"Epoch {epoch}",
            disable=not verbose,
            leave=False,
        ):
            batch = batch.to(device)
            total, edges, features = training_loss(model, batch, alpha)

            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            values = [total.item(), edges.item(), alpha * features.item()]
            history["batch_total"].append(values[0])
            history["batch_edges"].append(values[1])
            history["batch_nodes"].append(values[2])
            epoch_totals += torch.tensor(values, dtype=torch.float64)
            n_batches += 1

        train_losses = (epoch_totals / max(n_batches, 1)).tolist()
        validation_losses = evaluate(
            model,
            loader_validation,
            alpha,
            device,
            f"Validation {epoch}",
            verbose,
        )
        history["epoch_train"].append(train_losses)
        history["epoch_validation"].append(validation_losses)

        message = (
            f"Epoch {epoch}: train total {train_losses[0]:.5f} "
            f"(edges {train_losses[1]:.5f}, nodes {train_losses[2]:.5f})"
        )
        if validation_losses is not None:
            message += f", validation total {validation_losses[0]:.5f}"
        print(message)

        if experiment is not None:
            experiment.log_metric("train_total", train_losses[0], epoch=epoch)
            experiment.log_metric("train_edges", train_losses[1], epoch=epoch)
            experiment.log_metric("train_nodes", train_losses[2], epoch=epoch)
            if validation_losses is not None:
                experiment.log_metric(
                    "validation_total", validation_losses[0], epoch=epoch
                )

    print(f"--> Training completed in {time.time() - start:.2f} seconds")

    test_losses = evaluate(
        model, loader_testing, alpha, device, "Test", verbose
    )
    if test_losses is not None:
        history["test"] = test_losses
        print(f"Held-out test total loss: {test_losses[0]:.5f}")
        if experiment is not None:
            experiment.log_metric("test_total", test_losses[0])

    run_name = args.run_name or config.model.spec.replace(".", "_")
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_spec": config.model.spec,
            "model_kwargs": config.model.kwargs,
            "num_features": num_features,
            "means": means,
            "stds": stds,
            "split_state": split_state,
            "alpha": alpha,
            "graph": dict(config.graph),
            "training": settings,
            "train_samples": [sample.name for sample in samples],
        },
        checkpoint_path,
    )

    history_path = os.path.join(output_dir, "history.json")
    with open(history_path, "w") as history_file:
        json.dump(history, history_file, indent=2)

    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {history_path}")

    if experiment is not None:
        experiment.log_asset(checkpoint_path)
        experiment.log_asset(history_path)
        experiment.end()


if __name__ == "__main__":
    main()
