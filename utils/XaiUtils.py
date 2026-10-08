"""Explainable AI (XAI) utilities for Lund Graph Autoencoders.

Provides gradient-based feature attribution and per-node tree saliency methods
using pure PyTorch without external XAI dependencies.
"""

import numpy as np
import torch
import torch.nn.functional as F


def compute_node_feature_saliency(model, loader, device="cpu", max_batches=20, steps=5):
    """Compute global feature attribution using Integrated Gradients or Gradient x Input.

    Args:
        model: Graph Autoencoder model.
        loader: GraphDataLoader yielding PyG batches.
        device: Device string or torch.device.
        max_batches: Number of batches to process.
        steps: Number of Riemann steps for Integrated Gradients (steps=1 is Input x Grad).

    Returns:
        np.ndarray: 1D array of attribution scores per feature dimension.
    """
    model.eval()
    total_attribution = None
    total_nodes = 0

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = batch.to(device)
        x_target = batch.x.detach()
        baseline = torch.zeros_like(x_target)
        batch_grads = torch.zeros_like(x_target)

        for step in range(1, steps + 1):
            alpha = float(step) / steps
            x_step = (baseline + alpha * (x_target - baseline)).requires_grad_(True)
            if hasattr(model, "encoder") and hasattr(model, "features_decoder"):
                z = model.encoder(x_step, batch.edge_index)
                reconstructed = model.features_decoder(z, batch.edge_index)
            else:
                out = model(x_step, batch.edge_index)
                reconstructed = out[0] if isinstance(out, tuple) else out

            loss = F.mse_loss(reconstructed, x_step, reduction="sum")
            grad = torch.autograd.grad(loss, x_step)[0]
            batch_grads += grad.detach()

        avg_grads = batch_grads / steps
        integrated_attr = ((x_target - baseline) * avg_grads).abs().sum(dim=0).cpu().numpy()

        if total_attribution is None:
            total_attribution = np.zeros_like(integrated_attr)
        total_attribution += integrated_attr
        total_nodes += batch.x.shape[0]

    if total_attribution is None or total_nodes == 0:
        return np.array([])
    return total_attribution / total_nodes


def compute_tree_node_attributions(model, graph, device="cpu"):
    """Compute per-node reconstruction errors and gradient attributions for a single Lund graph.

    Args:
        model: Graph Autoencoder model.
        graph: PyG Data object representing one jet's Lund tree.
        device: Target execution device.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            (node_mse, node_saliency, edge_probs)
    """
    model.eval()
    x = graph.x.detach().to(device).requires_grad_(True)
    edge_index = graph.edge_index.to(device)

    if hasattr(model, "encoder") and hasattr(model, "features_decoder"):
        z = model.encoder(x, edge_index)
        reconstructed = model.features_decoder(z, edge_index)
    else:
        out = model(x, edge_index)
        reconstructed = out[0] if isinstance(out, tuple) else out
        z = getattr(model, "encoder", lambda _x, _e: None)(x, edge_index)

    node_mse = (x - reconstructed).square().mean(dim=1)
    loss = node_mse.sum()
    grad = torch.autograd.grad(loss, x)[0]
    node_saliency = (x * grad).abs().sum(dim=1).detach().cpu().numpy()

    edge_probs = np.array([])
    if hasattr(model, "edges_decoder") and z is not None and edge_index.shape[1] > 0:
        edge_probs = model.edges_decoder(z, edge_index, sigmoid=True).detach().cpu().numpy()

    return node_mse.detach().cpu().numpy(), node_saliency, edge_probs

