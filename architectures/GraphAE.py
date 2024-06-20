import torch
from torch_geometric.nn import GCNConv, GINConv, InnerProductDecoder

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
        # self.layers.append( GCNConv(num_features, architecture[0], improved=True) )
        self.layers.append( GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(num_features, num_features),
                torch.nn.ReLU(),
                torch.nn.Linear(num_features, architecture[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(architecture[0], architecture[0]),
                torch.nn.ReLU()
                )
            )
        )
        
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append( GINConv(
                torch.nn.Sequential(
                    torch.nn.Linear(in_dim, in_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_dim, out_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(out_dim, out_dim),
                    torch.nn.ReLU()
                    )
                ) 
            )
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index).relu()
        return self.layers[-1](x, edge_index)

class features_decoder(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(features_decoder, self).__init__()
        
        # Sequential model to be passed as nn parameter to GINConv
        self.gin_model = torch.nn.Sequential(
            torch.nn.Linear(architecture[-1], architecture[-1]),
            torch.nn.ReLU(),
            torch.nn.Linear(architecture[-1], num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_features),
            torch.nn.ReLU()
        )
        
        self.decoder = GINConv(self.gin_model, eps=0.0)

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