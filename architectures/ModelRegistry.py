"""Build models from a configuration string, so the pipeline never names one.

A model is identified by a ``"Module.ClassName"`` spec resolved inside this
package, for example ``"GraphAE.graphAE"`` or ``"GraphTransformerAE.
graphTransformerAE"``. A new architecture therefore only needs a new file in
``architectures/`` and a one line change in ``gae_config.json``; no pipeline
code has to be touched.

A model must provide:

- ``forward(x, edge_index)``
- ``score_batch(x, edge_index, batch, alpha)``, which is all that
  ``RootGraphScorer`` needs for inference
- either ``training_loss(batch, alpha)``, returning
  ``(total, edge_term, feature_term)``, or both ``edges_loss`` and
  ``features_loss`` following the ``graphAE`` convention

Models that define ``training_loss`` control their own objective. Everything
else falls back to the autoencoder loss used in ``example.ipynb``.
"""

import importlib

from torch_geometric.utils import batched_negative_sampling


MODEL_CONTRACT = (
    "forward(x, edge_index); "
    "score_batch(x, edge_index, batch, alpha); "
    "training_loss(batch, alpha) or both edges_loss and features_loss"
)


def build_model(spec, num_features, **kwargs):
    """Instantiate the architecture named by *spec*.

    Args:
        spec (str): ``"Module.ClassName"`` resolved under ``architectures/``.
        num_features (int): Number of node features of the input graphs.
        **kwargs: Constructor arguments forwarded to the architecture, so a
            new model with different arguments needs no change here.
    """
    if spec.count(".") != 1:
        raise ValueError(
            f"Model spec {spec!r} must be 'Module.ClassName', for example "
            "'GraphAE.graphAE'"
        )
    module_name, class_name = spec.split(".")

    try:
        module = importlib.import_module(f"architectures.{module_name}")
    except ImportError as error:
        raise ImportError(
            f"Could not import 'architectures.{module_name}' for model spec "
            f"{spec!r}"
        ) from error

    if not hasattr(module, class_name):
        raise AttributeError(
            f"'architectures.{module_name}' has no attribute {class_name!r} "
            f"for model spec {spec!r}"
        )

    model = getattr(module, class_name)(num_features=num_features, **kwargs)
    validate_model(model, spec)
    return model


def validate_model(model, spec="model"):
    """Raise if *model* does not implement the documented contract."""
    missing = [
        method
        for method in ("forward", "score_batch")
        if not callable(getattr(model, method, None))
    ]
    if not callable(getattr(model, "training_loss", None)):
        missing.extend(
            method
            for method in ("edges_loss", "features_loss")
            if not callable(getattr(model, method, None))
        )
    if missing:
        raise TypeError(
            f"{spec} is missing {sorted(set(missing))}. "
            f"A model must provide: {MODEL_CONTRACT}"
        )
    return model


def training_loss(model, batch, alpha=1.0):
    """Return ``(total, edge_term, feature_term)`` for one batch.

    Uses the model's own ``training_loss`` when it defines one, otherwise the
    autoencoder loss ``edges_loss + alpha * features_loss`` with batched
    negative sampling, as in ``example.ipynb``.
    """
    if callable(getattr(model, "training_loss", None)):
        return model.training_loss(batch, alpha)

    negative_edges = batched_negative_sampling(batch.edge_index, batch.batch)
    edges_loss = model.edges_loss(batch.x, batch.edge_index, negative_edges)
    features_loss = model.features_loss(batch.x, batch.edge_index)
    return edges_loss + alpha * features_loss, edges_loss, features_loss
