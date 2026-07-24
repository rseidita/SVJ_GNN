import torch
from torch_geometric.nn import GCNConv, InnerProductDecoder

EPS = 1e-15

class weighted_MSELoss(torch.nn.Module):
    def __init__(self):
        super(weighted_MSELoss, self).__init__()
    def forward(self, x, y, w):
        return (x-y).pow(2).sum(dim=1).mul(w).sum()/w.sum()

class encoder(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append( GCNConv(num_features, architecture[0], improved=True) )
        
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append( GCNConv(in_dim, out_dim, improved=True) )
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index).relu()
        return self.layers[-1](x, edge_index)

class features_decoder(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(features_decoder, self).__init__()
        self.decoder = GCNConv(architecture[-1], num_features, improved=True)
        
    def forward(self, z, edge_index):
        return self.decoder(z, edge_index)
    
class graphAE(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(graphAE, self).__init__()
        self.encoder = encoder(num_features, architecture)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoder(num_features, architecture)
        
    def edges_loss(self, x, edge_index, negative_edge_index, weight=None):
        
        z = self.encoder(x, edge_index)
        
        if weight is None:
            # weight = torch.ones(edge_index.shape[1]).to(x.device)
            weight = 1.

        pos_loss = (-torch.log(
                self.edges_decoder(z, edge_index, sigmoid=True) + EPS
                )*weight
            ).mean()
        neg_loss = 0.0
        if self.training:
            neg_loss = (
                -torch.log(
                    1 - self.edges_decoder(z, negative_edge_index, sigmoid=True) + EPS
                    )*weight
                ).mean()
        
        return (pos_loss + neg_loss)
    
    def features_loss(self, x, edge_index, weight=None):

        z = self.encoder(x, edge_index)
        x_reco = self.features_decoder(z, edge_index)
        return (weighted_MSELoss()(x, x_reco, weight)) if weight is not None else torch.nn.MSELoss()(x, x_reco)

    def score_batch(self, x, edge_index, batch, alpha=1.0):
        """Return total, edge, and feature reconstruction scores per graph.

        Inference scores use observed (positive) edges only. Negative edges
        prevent the trivial all-positive solution during training, but are not
        currently part of the anomaly score. This choice can be revisited if
        false-positive edge predictions prove useful at inference time.
        """
        if batch.ndim != 1 or len(batch) != len(x):
            raise ValueError("batch must identify the graph of every node")
        if len(batch) == 0:
            empty = x.new_empty(0)
            return empty, empty, empty

        n_graphs = int(batch.max().item()) + 1
        z = self.encoder(x, edge_index)
        x_reco = self.features_decoder(z, edge_index)

        node_error = (x - x_reco).square().mean(dim=1)
        feature_sum = x.new_zeros(n_graphs)
        feature_sum.index_add_(0, batch, node_error)
        node_count = torch.bincount(batch, minlength=n_graphs).clamp_min(1)
        feature_score = feature_sum / node_count

        edge_score = x.new_zeros(n_graphs)
        if edge_index.shape[1]:
            edge_probability = self.edges_decoder(
                z, edge_index, sigmoid=True
            )
            edge_error = -torch.log(edge_probability + EPS)
            edge_graph = batch[edge_index[0]]
            edge_score.index_add_(0, edge_graph, edge_error)
            edge_count = torch.bincount(
                edge_graph, minlength=n_graphs
            ).clamp_min(1)
            edge_score = edge_score / edge_count

        total_score = edge_score + alpha * feature_score
        return total_score, edge_score, feature_score

    def score_event(self, graph_or_x, edge_index=None, alpha=1.0):
        """Return total, edge, and feature scores for one graph.

        ``graph_or_x`` may be a PyG ``Data`` object, such as ``dataset[0]``,
        or its node-feature tensor when ``edge_index`` is passed separately.
        """
        if edge_index is None:
            if not (
                hasattr(graph_or_x, "x")
                and hasattr(graph_or_x, "edge_index")
            ):
                raise TypeError(
                    "Pass a graph with x and edge_index attributes, or pass "
                    "x and edge_index separately"
                )
            x = graph_or_x.x
            edge_index = graph_or_x.edge_index
        else:
            x = graph_or_x

        batch = torch.zeros(
            len(x), dtype=torch.long, device=x.device
        )
        total, edge, feature = self.score_batch(
            x, edge_index, batch, alpha=alpha
        )
        return total[0], edge[0], feature[0]
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.features_decoder(z, edge_index), self.edges_decoder(z, edge_index, sigmoid=True)
