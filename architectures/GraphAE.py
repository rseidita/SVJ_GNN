import torch
from torch_geometric.nn import GCNConv, InnerProductDecoder
import torch.nn.functional as F
import torch.nn.init as init  # Import for initialization

EPS = 1e-15

def dropout_edges(edge_index, p=0.1):
    # Randomly drop a percentage of edges
    edge_index, _ = utils.dropout_adj(edge_index, p=p, force_undirected=True, training=True)
    return edge_index


class weighted_MSELoss(torch.nn.Module):
    def __init__(self):
        super(weighted_MSELoss, self).__init__()
    def forward(self, x, y, w):
        return (x-y).pow(2).sum(dim=1).mul(w).sum()/w.sum()

class encoder(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(encoder, self).__init__()
        self.dropout = dropout
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
            x = self.batch_norms[i](x)  # Apply BatchNorm
            x = F.relu(x)  # Apply activation function
        
         #Last layer (without BatchNorm)
        return self.layers[-1](x, edge_index)

class features_decoder(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(features_decoder, self).__init__()
        self.dropout=dropout
        self.decoder = GCNConv(architecture[-1], num_features, improved=True)
        '''self.layersx = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for in_dim, out_dim in zip(back_architecture[:-1], back_architecture[1:]):
            self.layersx.append( GCNConv(in_dim, out_dim, improved=True) )
            self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))
        self.layersx.append( GCNConv(back_architecture[-1], num_features , improved=True) )



        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.layersx:
            if isinstance(layer, GCNConv):
                init.kaiming_uniform_(layer.lin.weight, nonlinearity='relu')'''
        
    def forward(self, z, edge_index):
        '''for i, layer in enumerate(self.layersx[:-1]):
            z = layer(z, edge_index)
            z = self.batch_norms[i](z)  # Apply BatchNorm after GCN
            z = F.leaky_relu(z,negative_slope=0.01)
        return self.layersx[-1](z, edge_index)'''
        z = F.dropout(z, p=self.dropout, training=self.training)
        return self.decoder(z, edge_index)
    
class graphAE(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(graphAE, self).__init__()
        self.encoder = encoder(num_features, architecture, dropout)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoder(num_features, architecture, dropout)
        #self.dropout=0.15
        #self.edge_dropout = 0.2
        
    def edges_loss(self, x, edge_index, negative_edge_index, weight=None):
        
        #edge_index = dropout_edges(edge_index, p=self.edge_dropout) 
        z = self.encoder(x, edge_index)
        #z = F.dropout(z, p=self.dropout, training=self.training)
        
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
        neg_loss=0.0
        if (negative_edge_index is not None and negative_edge_index.numel()> 0):
            neg_loss = (-torch.log(torch.clamp(1 - self.edges_decoder(z, negative_edge_index, sigmoid=True), min=EPS, max=1 - EPS)) * weight).mean()

        
        return (pos_loss + neg_loss)
    
    def features_loss(self, x, edge_index, weight=None):

        #edge_index = dropout_edges(edge_index, p=self.edge_dropout) 
        z = self.encoder(x, edge_index)
        #z = F.dropout(z, p=self.dropout, training=self.training)
        x_reco = self.features_decoder(z, edge_index)
        return (weighted_MSELoss()(x, x_reco, weight)) if weight is not None else torch.nn.MSELoss()(x, x_reco)
    
    def forward(self, x, edge_index):
        
        #edge_index = dropout_edges(edge_index, p=self.edge_dropout) 
        z = self.encoder(x, edge_index)
        #z = F.dropout(z, p=self.dropout, training=self.training)
        return self.features_decoder(z, edge_index), self.edges_decoder(z, edge_index, sigmoid=True)