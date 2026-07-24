import torch
from torch_geometric.nn import TransformerConv

from architectures.GraphAE import graphAE


class TransformerEncoder(torch.nn.Module):
    """Graph-transformer encoder with fixed output width per attention head."""

    def __init__(self, num_features, architecture, heads, dropout):
        super().__init__()
        dimensions = (num_features, *architecture)
        self.layers = torch.nn.ModuleList([
            TransformerConv(
                in_channels,
                out_channels,
                heads=heads,
                concat=False,
                beta=True,
                dropout=dropout,
            )
            for in_channels, out_channels in zip(
                dimensions[:-1], dimensions[1:]
            )
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(width) for width in architecture[:-1]
        ])

    def forward(self, x, edge_index):
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = norm(layer(x, edge_index)).relu()
        return self.layers[-1](x, edge_index)


class TransformerFeatureDecoder(torch.nn.Module):
    """Mirror the encoder widths and reconstruct node features."""

    def __init__(self, num_features, architecture, heads, dropout):
        super().__init__()
        dimensions = (
            architecture[-1],
            *reversed(architecture[:-1]),
            num_features,
        )
        self.layers = torch.nn.ModuleList([
            TransformerConv(
                in_channels,
                out_channels,
                heads=heads,
                concat=False,
                beta=True,
                dropout=dropout,
            )
            for in_channels, out_channels in zip(
                dimensions[:-1], dimensions[1:]
            )
        ])
        self.norms = torch.nn.ModuleList([
            torch.nn.LayerNorm(width) for width in dimensions[1:-1]
        ])

    def forward(self, z, edge_index):
        x = z
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = norm(layer(x, edge_index)).relu()
        return self.layers[-1](x, edge_index)


class GraphTransformerAE(graphAE):
    """Small attention-based alternative to the GCN graph autoencoder.

    It retains the complete ``graphAE`` loss and scoring interface, allowing
    it to be used by the same training loops and ``RootGraphScorer``.
    """

    def __init__(
        self,
        num_features,
        architecture=(64, 32, 16),
        *,
        heads=2,
        dropout=0.0,
    ):
        if not architecture or any(width <= 0 for width in architecture):
            raise ValueError("architecture must contain positive widths")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        # Initialize the common decoders and scoring interface, then replace
        # the GCN feature path with attention-based encoder/decoder modules.
        super().__init__(num_features, architecture)
        self.encoder = TransformerEncoder(
            num_features, architecture, heads, dropout
        )
        self.features_decoder = TransformerFeatureDecoder(
            num_features, architecture, heads, dropout
        )
        self.architecture = tuple(architecture)
        self.heads = int(heads)
        self.dropout = float(dropout)


# Match the existing lower-case ``graphAE`` naming style while also exposing
# the PEP 8 class name for new code.
graphTransformerAE = GraphTransformerAE
