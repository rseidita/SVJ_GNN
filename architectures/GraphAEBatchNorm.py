import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, InnerProductDecoder
from torch_geometric.utils import batched_negative_sampling

EPS = 1e-15


class EncoderBatchNorm(torch.nn.Module):
    """GCN encoder with batch normalization, matching Aimar's checkpoint structure."""

    def __init__(self, num_features: int, architecture, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        dimensions = (num_features, *architecture)
        self.layers = torch.nn.ModuleList(
            GCNConv(in_dim, out_dim, improved=True)
            for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])
        )
        # Linear projections registered for strict state_dict compatibility
        self.linear_projections = torch.nn.ModuleList(
            torch.nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(dimensions[:-1], dimensions[1:])
        )
        self.batch_norms = torch.nn.ModuleList(
            torch.nn.BatchNorm1d(out_dim) for out_dim in architecture[:-1]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for index, layer in enumerate(self.layers[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index, edge_weight=edge_weight)
            x = F.relu(self.batch_norms[index](x))
        return self.layers[-1](x, edge_index, edge_weight=edge_weight)


class FeaturesDecoderBatchNorm(torch.nn.Module):
    """Direct single-layer GCN feature decoder from latent representation."""

    def __init__(self, num_features: int, architecture, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.decoder = GCNConv(architecture[-1], num_features, improved=True)

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.decoder(z, edge_index, edge_weight=edge_weight)


class GraphAEBatchNorm(torch.nn.Module):
    """Graph autoencoder with BatchNorm matching Aimar's architecture and SVJ_GNN contract."""

    def __init__(self, num_features: int = 12, architecture=(115, 170, 200, 115), dropout: float = 0.1, negative_sampler: str = "gpu"):
        super().__init__()
        self.encoder = EncoderBatchNorm(num_features, architecture, dropout)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = FeaturesDecoderBatchNorm(num_features, architecture, dropout)
        self.register_buffer("center", torch.zeros(architecture[-1]))
        self.architecture = tuple(architecture)
        self.num_features = num_features
        self.dropout = dropout
        self.negative_sampler = negative_sampler

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, edge_index, edge_weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, edge_index, edge_weight)
        reconstructed_x = self.features_decoder(z, edge_index, edge_weight)
        edge_probability = self.edges_decoder(z, edge_index, sigmoid=True)
        return reconstructed_x, edge_probability

    def features_loss(self, x, edge_index, weight=None, z=None):
        if z is None:
            z = self.encoder(x, edge_index)
        reconstructed_x = self.features_decoder(z, edge_index)
        if weight is None:
            return F.mse_loss(reconstructed_x, x)
        per_node = (x - reconstructed_x).square().sum(dim=1)
        return (per_node * weight).sum() / weight.sum()

    def edges_loss(self, x, edge_index, negative_edge_index, weight=None, z=None):
        if z is None:
            z = self.encoder(x, edge_index)
        loss_weight = 1.0 if weight is None else weight
        if edge_index.shape[1] == 0:
            positive_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
        else:
            probability = self.edges_decoder(z, edge_index, sigmoid=True)
            positive_loss = (-torch.log(probability.clamp(EPS, 1.0 - EPS)) * loss_weight).mean()

        negative_loss = torch.tensor(0.0, device=x.device)
        if self.training and negative_edge_index is not None and negative_edge_index.numel() > 0:
            negative_probability = self.edges_decoder(z, negative_edge_index, sigmoid=True)
            negative_loss = (-torch.log((1.0 - negative_probability).clamp(EPS, 1.0 - EPS)) * loss_weight).mean()

        return positive_loss + negative_loss

    def _sample_negative_edges_gpu(self, edge_index, batch):
        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return edge_index.new_empty((2, 0))
        n_graphs = int(batch[-1].item()) + 1 if batch.numel() > 0 else 0
        node_counts = torch.bincount(batch, minlength=n_graphs)
        ptr = torch.cat([batch.new_zeros(1), torch.cumsum(node_counts, dim=0)])
        edge_graph = batch[edge_index[0]]
        g_start = ptr[edge_graph]
        g_len = node_counts[edge_graph]
        u_offset = torch.randint(0, 1000000, (num_edges,), device=edge_index.device) % g_len
        delta = 1 + (torch.randint(0, 1000000, (num_edges,), device=edge_index.device) % (g_len - 1).clamp_min(1))
        v_offset = (u_offset + delta) % g_len
        rand_src = g_start + u_offset
        rand_dst = g_start + v_offset
        return torch.stack([rand_src, rand_dst], dim=0)

    def training_loss(self, batch, alpha=1.0):
        z = self.encoder(batch.x, batch.edge_index)
        if getattr(self, "negative_sampler", "gpu") == "cpu":
            negative_edges = batched_negative_sampling(batch.edge_index, batch.batch)
        elif not self.training:
            negative_edges = None
        else:
            negative_edges = self._sample_negative_edges_gpu(batch.edge_index, batch.batch)
        edges = self.edges_loss(batch.x, batch.edge_index, negative_edges, z=z)
        features = self.features_loss(batch.x, batch.edge_index, z=z)
        return edges + alpha * features, edges, features

    def score_batch(self, x, edge_index, batch, alpha=1.0):
        """Return total, edge, and feature scores for every graph in a PyG batch."""
        if batch.ndim != 1 or len(batch) != len(x):
            raise ValueError("batch must identify the graph of every node")
        if len(batch) == 0:
            empty = x.new_empty(0)
            return empty, empty, empty

        n_graphs = int(batch.max().item()) + 1
        z = self.encoder(x, edge_index)
        reconstructed_x = self.features_decoder(z, edge_index)
        node_error = (x - reconstructed_x).square().mean(dim=1)
        feature_sum = x.new_zeros(n_graphs)
        feature_sum.index_add_(0, batch, node_error)
        node_count = torch.bincount(batch, minlength=n_graphs).clamp_min(1)
        feature_score = feature_sum / node_count

        edge_score = x.new_zeros(n_graphs)
        if edge_index.shape[1]:
            probability = self.edges_decoder(z, edge_index, sigmoid=True)
            edge_error = -torch.log(probability.clamp(EPS, 1.0 - EPS))
            edge_graph = batch[edge_index[0]]
            edge_score.index_add_(0, edge_graph, edge_error)
            edge_count = torch.bincount(edge_graph, minlength=n_graphs).clamp_min(1)
            edge_score = edge_score / edge_count

        total_score = edge_score + alpha * feature_score
        return total_score, edge_score, feature_score

    def score_event(self, graph_or_x, edge_index=None, alpha=1.0):
        """Score one PyG graph or an explicit x/edge_index tensor pair."""
        if edge_index is None:
            if not hasattr(graph_or_x, "x") or not hasattr(graph_or_x, "edge_index"):
                raise TypeError("Pass a graph with x and edge_index, or both tensors")
            x, edge_index = graph_or_x.x, graph_or_x.edge_index
        else:
            x = graph_or_x
        batch = torch.zeros(len(x), dtype=torch.long, device=x.device)
        total, edge, feature = self.score_batch(x, edge_index, batch, alpha=alpha)
        return total[0], edge[0], feature[0]

