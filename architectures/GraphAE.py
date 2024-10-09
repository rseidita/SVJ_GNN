import torch
from torch_geometric.nn import GCNConv, InnerProductDecoder
import torch.nn.functional as F
import torch.nn.init as init  # Import for initialization

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
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)  # Apply GCN
            x = F.relu(x)  # Apply activation function
        
         #Last layer (without BatchNorm)
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
            weight = 1.00
        
        pos_loss = (
            -torch.log(
                self.edges_decoder(z, edge_index, sigmoid=True) + EPS
            ) * weight).mean()

        neg_loss = (-torch.log(1 - self.edges_decoder(z, negative_edge_index, sigmoid=True)+ EPS) * weight).mean()

        
        return (pos_loss + neg_loss)
    
    def features_loss(self, x, edge_index, weight=None):

        z = self.encoder(x, edge_index)
        x_reco = self.features_decoder(z, edge_index)
        return (weighted_MSELoss()(x, x_reco, weight)) if weight is not None else torch.nn.MSELoss()(x, x_reco)
    
    def forward(self, x, edge_index):
        
        z = self.encoder(x, edge_index)
        return self.features_decoder(z, edge_index), self.edges_decoder(z, edge_index, sigmoid=True)
    




class encoderBatchNorm(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout, batch_norm):
        super(encoderBatchNorm, self).__init__()
        self.dropout = dropout
        self.batchn_norm=batch_norm
        self.layers = torch.nn.ModuleList()
        self.layers.append( GCNConv(num_features, architecture[0], improved=True) )
        
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append( GCNConv(in_dim, out_dim, improved=True) )
        

        self.batch_norms = torch.nn.ModuleList()
        for out_dim in architecture[:-1]:  # BatchNorm for all layers except the last
           self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

        self._initialize_weights()

    def _initialize_weights(self):
       for layer in self.layers:
            if isinstance(layer, GCNConv):
                init.kaiming_uniform_(layer.lin.weight, nonlinearity='relu')

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x, edge_index)  # Apply GCN
            if self.batchn_norm:
                x = self.batch_norms[i](x)  # Apply BatchNorm
            x = F.relu(x)  # Apply activation function
        
         #Last layer (without BatchNorm)
        return self.layers[-1](x, edge_index)

class features_decoderBatchNorm(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(features_decoderBatchNorm, self).__init__()
        self.dropout=dropout
        self.decoder = GCNConv(architecture[-1], num_features, improved=True)
    
    def forward(self, z, edge_index):
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.decoder(z, edge_index)
    
class graphAEBatchNorm(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout, batch_norm):
        super(graphAEBatchNorm, self).__init__()
        self.encoder = encoderBatchNorm(num_features, architecture, dropout, batch_norm)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoderBatchNorm(num_features, architecture, dropout)
        
    def edges_loss(self, x, edge_index, negative_edge_index, weight=None):
        
        z = self.encoder(x, edge_index)
        
        if weight is None:
            # weight = torch.ones(edge_index.shape[1]).to(x.device)
            weight = 1.00
        
        assert negative_edge_index is not None, "Negative edge index is None"
        assert negative_edge_index.numel() > 0, "Negative edge index is empty"
        assert not torch.isnan(negative_edge_index).any(), "Negative edge index contains NaNs"
        assert not torch.isinf(negative_edge_index).any(), "Negative edge index contains infinities"
        
        pos_loss = (
            -torch.log(
                torch.clamp(self.edges_decoder(z, edge_index, sigmoid=True), min=EPS, max=1-EPS)
            ) * weight).mean()

        # For neg_loss
        neg_loss = (-torch.log(torch.clamp(1 - self.edges_decoder(z, negative_edge_index, sigmoid=True), min=EPS, max=1 - EPS)) * weight).mean()

        
        return (pos_loss + neg_loss)
    
    def features_loss(self, x, edge_index, weight=None):

        z = self.encoder(x, edge_index)
        x_reco = self.features_decoder(z, edge_index)
        return (weighted_MSELoss()(x, x_reco, weight)) if weight is not None else torch.nn.MSELoss()(x, x_reco)
    
    def forward(self, x, edge_index):
        
        z = self.encoder(x, edge_index)
        return self.features_decoder(z, edge_index), self.edges_decoder(z, edge_index, sigmoid=True)