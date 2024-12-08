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
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.features_decoder(z, edge_index), self.edges_decoder(z, edge_index, sigmoid=True)