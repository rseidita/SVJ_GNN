import torch
from torch_geometric.nn import GCNConv, InnerProductDecoder, GATConv, GINConv
import torch.nn.functional as F
import torch.nn.init as init  # Import for initialization
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import negative_sampling, batched_negative_sampling
import numpy as np

EPS = 1e-15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        
        #self.linear= torch.nn.Linear(num_features, 8)
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append( GCNConv(in_dim, out_dim, improved=True) )
        
    def forward(self, x, edge_index):
        
        #x = self.linear(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index)  # Apply GCN
            x = F.relu(x) +x  # Apply activation function
        
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
    def __init__(self, num_features, architecture, dropout):
        super(encoderBatchNorm, self).__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.layers.append( GCNConv(num_features, architecture[0], improved=True) )
        self.linear_projections = torch.nn.ModuleList()
        
        self.linear_projections.append(torch.nn.Linear(num_features, architecture[0] ))
        
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append( GCNConv(in_dim, out_dim, improved=True) )
            self.linear_projections.append(torch.nn.Linear(in_dim, out_dim))
        

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
            t = x
            x = layer(x, edge_index)  # Apply GCN
            residual = self.linear_projections[i](t)
            x = self.batch_norms[i](x)  # Apply BatchNorm
            x = F.relu(x) + residual  # Apply activation function
        
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
    def __init__(self, num_features, architecture, dropout):
        super(graphAEBatchNorm, self).__init__()
        self.encoder = encoderBatchNorm(num_features, architecture, dropout)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoderBatchNorm(num_features, architecture, dropout)
        
    def edges_loss(self, x, edge_index, negative_edge_index, weight=None):
        
        z = self.encoder(x, edge_index)
        
        if weight is None:
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

        z = self.encoder(x, edge_index)
        x_reco = self.features_decoder(z, edge_index)
        return (weighted_MSELoss()(x, x_reco, weight)) if weight is not None else torch.nn.MSELoss()(x, x_reco)
    
    def forward(self, x, edge_index):
        
        z = self.encoderBatchNorm(x, edge_index)
        return self.features_decoder(z, edge_index), self.edges_decoder(z, edge_index, sigmoid=True)




class encoderBatchNormGraphPool(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(encoderBatchNormGraphPool, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append( GCNConv(num_features, architecture[0], improved=False) )
        self.GraphPool=GraphPool(num_features, architecture)
        self.dropout = dropout
        
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append( GCNConv(in_dim, out_dim, improved=False) )
        self.layers.append(GCNConv(out_dim, out_dim))
        

        self.batch_norms = torch.nn.ModuleList()
        for out_dim in architecture[:-1]:  # BatchNorm for all layers except the last
           self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))

        self._initialize_weights()

    def _initialize_weights(self):
       for layer in self.layers:
            if isinstance(layer, GCNConv):
                init.kaiming_uniform_(layer.lin.weight, nonlinearity='relu')

    def forward(self, x, edge_index):
        x_list=[]
        node_indices=[]
        edges=[]
        for i, layer in enumerate(self.layers[:-2]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            edges.append(edge_index)
            x = layer(x, edge_index)
            x_list.append(x)
            x , edge_index, indices = self.GraphPool(x, edge_index)
            #x = self.batch_norms[i](x)
            x = F.relu(x)
            node_indices.append(indices)
        
         #Last layer (without BatchNorm)
        x = self.layers[-2](x, edge_index)
        x_list.append(x)
        x, edge_index, indices = self.GraphPool(x, edge_index)
        edges.append(edge_index)
        node_indices.append(indices)
        x = self.layers[-1](x, edge_index)
        
        return x, edges, node_indices, x_list #x_ten, x_list, edges, node_indices, x

class features_decoderBatchNormGraphPool(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(features_decoderBatchNormGraphPool, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.GraphUnpool=GraphUnpool(num_features, architecture)
        self.scalar = torch.nn.Parameter(torch.randn(1))
        self.constrained_scalar=F.sigmoid(self.scalar)
        self.dropout = dropout

        back_arch=architecture[::-1]

        for in_dim, out_dim in zip(back_arch[:-1], back_arch[1:]):
            self.layers.append( GCNConv(in_dim, out_dim, improved=True) )
        
        self.layers.append(GCNConv(architecture[0], num_features, improved=True) )
        
        self.batch_norms = torch.nn.ModuleList()
        for out_dim in architecture[:-1]:  # BatchNorm for all layers except the last
           self.batch_norms.append(torch.nn.BatchNorm1d(out_dim))
        
        
    def forward(self, z, edge_indexz, node_indices, x_list):
        for i, layer in enumerate(self.layers[:-1]):
            
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = self.GraphUnpool(z ,  node_indices[-1-i],  x_list[-1-i])
            z = z + x_list[-i-1]#*self.constrained_scalar
            z = layer(z, edge_indexz[-i-1])  # Apply GCN
            z = self.batch_norms[i](z)  # Apply BatchNorm
            z = F.relu(z)  # Apply activation function
            
        z = self.GraphUnpool(z ,  node_indices[0], x_list[0])
        z = z + x_list[0]#*self.constrained_scalar*x_list[-i-1]
        
        return self.layers[-1](z, edge_indexz[0]), edge_indexz[0] , z
    
class graphAEBatchNormGraphPool(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(graphAEBatchNormGraphPool, self).__init__()
        self.encoder = encoderBatchNormGraphPool(num_features, architecture, dropout)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoderBatchNormGraphPool(num_features, architecture, dropout)
        
    def edges_loss(self, x, edge_index, negative_edge_index, weight=None, batched=False):
        
        z, edge_indexz, node_indices, x_list= self.encoder(x, edge_index)
        
        x_reco, edge_index, z= self.features_decoder(z, edge_indexz, node_indices, x_list)
        #z_ten, z_list, edges, node_indices, x = self.encoder(x, edge_index)
        
        if weight is None:
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

        z , edge_indexz, node_indices, x_list = self.encoder(x, edge_index)
        #z_ten, z_list, edges, node_indices, x = self.encoder(x, edge_index)
        x_reco , edge_index, z = self.features_decoder(z, edge_indexz, node_indices, x_list)
        return (weighted_MSELoss()(x, x_reco, weight)) if weight is not None else torch.nn.MSELoss()(x, x_reco)
    
    def latent_space_loss(self, x, edge_index, value, j):
        
        z , edge_indexz, node_indices, x_list = self.encoder(x, edge_index)
        
        n_feat = z.shape[1]
        
        if j==1:
            
            #fro_norm = torch.norm(z, p="fro")
            node_avg = torch.mean(z, dim=0)  # Shape becomes (n_feat,)
            
            #l2_norm = torch.norm(node_avg, p=2)

            # Step 2: Average over the feature dimension (dim=0)
            index = np.argsort(node_avg.detach().cpu().numpy())[-1:] # get the the index of the highest value
            highest_value = node_avg[index]
            #global_avg = torch.mean(node_avg)
            
            return (highest_value-value)**2 #l2_norm, fro_norm, highest_value
        if j==2:
            
            node_avg = torch.mean(z, dim=0)
            linear_layer = torch.nn.Linear(n_feat, 2)  # Map from n_feat to 2
            x_2d_vector = linear_layer(node_avg)  # Shape becomes (2,)

            return(x_2d_vector-value)**2
        #another option
        '''pos_loss = (
            -torch.log(
                torch.clamp(self.edges_decoder(z, edge_indexz, sigmoid=True), min=EPS, max=1-EPS)
            )).mean()

        # For neg_loss
        
        neg_loss = (-torch.log(torch.clamp(1 - self.edges_decoder(z, negative_sampling(edge_indexz), sigmoid=True), min=EPS, max=1 - EPS)) * weight).mean() 
        return pos_loss + neg_loss  '''
        
    def forward(self, x, edge_index):
        
        z , edge_indexz, node_indices, x_list = self.encoder(x, edge_index)
        #z_ten, z_list, edges, node_indices, x = self.encoder(x, edge_index)
        return self.features_decoder(z, edge_indexz, node_indices, x_list), self.edges_decoder(z, edge_indexz[-1], sigmoid=True)





class GraphPool(torch.nn.Module):  # Inherit from torch.nn.Module
    def __init__(self, num_features, architecture):
        super(GraphPool, self).__init__()  # Call parent constructor
        self.num_features = num_features
        self.p = None

    def forward(self, x, edge_index):# num_feat, xx, edge_indexxx, m):
        
        edge_index_out = []
        
        indices_together=[]
        
        '''parameter = torch.nn.Parameter(torch.randn( x.shape[1]))
        normalized_parameter = F.normalize(parameter, p=2, dim=0)
        projected = torch.matmul(x, normalized_parameter)
        projected = F.sigmoid(projected)'''
        dense_edge=to_dense_adj(edge_index)
        dense_edge = dense_edge[0]
        dense_edge, indices= self.remove_leaf_nodes(dense_edge)
        '''projected = projected.view(-1, 1)
        x = x * projected'''
        x_sample = x[indices]
        indices_together.append(indices)
        edge_index_out, sparse_edge_attr = dense_to_sparse(dense_edge)
        
        return x_sample, edge_index_out, indices_together

        
    def remove_leaf_nodes(self, matrix):
        
        
        counts = matrix.sum(dim=1)  # Sum over rows to count connections for each node

        indices = (counts > 1).nonzero(as_tuple=True)[0]  # Non-leaf node indices

        # If the number of non-leaf nodes is <= 2, return the original matrix (no removal)
        if indices.numel() <= 2:
            return matrix, torch.arange(matrix.size(0))  # Return all nodes, keeping the original matrix

        matrix = matrix[indices][:, indices]  # Filter rows and columns for non-leaf nodes

        return matrix, indices


class GraphUnpool(torch.nn.Module):  # Inherit from torch.nn.Module
    def __init__(self, num_features, architecture):
        super(GraphUnpool, self).__init__()  # Call parent constructor
        self.num_features = num_features

    def forward(self, z, node_indices, xx_list):  # Include 'self'
        target_nodes = []
        # Create target tensor with the same number of rows as xx_list[i] and num_features columns
        unpooled_tensor = torch.zeros_like(xx_list)
        node_indices_ten = node_indices[0]
        unpooled_tensor[node_indices_ten] = z

        return  unpooled_tensor




import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EncoderAttention(torch.nn.Module):
    def __init__(self, num_features, architecture, dropout):
        super(EncoderAttention, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.linear_projections = torch.nn.ModuleList()
        self.dropout = dropout

        # Initial layer setup
        self.layers.append(GATConv(num_features, architecture[0], heads=2, dropout=dropout, concat=False))
        self.linear_projections.append(torch.nn.Linear(num_features, architecture[0] ))

        # Intermediate layers setup
        for in_dim, out_dim in zip(architecture[:-1], architecture[1:]):
            self.layers.append(GATConv(in_dim, out_dim, heads=2, dropout=dropout, concat=False))
            
            # Projection to match dimensions
            self.linear_projections.append(torch.nn.Linear(in_dim , out_dim ))

        # Batch normalization for all but the last layer
        self.batch_norms = torch.nn.ModuleList()
        for out_dim in architecture[:-1]:
            self.batch_norms.append(torch.nn.BatchNorm1d(out_dim ))  # Adjust for concatenated output size



    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.dropout(x, p=self.dropout, training=self.training)
            t = x  # Save the input for the residual connection
            x = layer(x, edge_index)  # Apply GAT layer
            x = F.relu(x)  # Apply activation function

            if self.linear_projections[i]:  # If there's a linear projection for the residual
                residual = self.linear_projections[i](t)  # Project input to match output dimension
            else:
                residual = t  # If no projection needed, use original input

            x = x + residual  # Add residual connection
            
            if i < len(self.batch_norms):  # Apply BatchNorm if not the last layer
                x = self.batch_norms[i](x)

        # Last layer (without BatchNorm)
        return self.layers[-1](x, edge_index)  # Last GAT layer without residual connection


class features_decoderAttention(torch.nn.Module):
    def __init__(self, num_features, back_architecture, dropout):
        super(features_decoderAttention, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dropout=dropout
        self.linear_projections = torch.nn.ModuleList()


        for in_dim, out_dim in zip(back_architecture[:-1], back_architecture[1:]):
            self.layers.append( GATConv(in_dim, out_dim, heads=2, dropout=0.3, concat=False) )

            if in_dim != out_dim:
                self.linear_projections.append(torch.nn.Linear(in_dim, out_dim))
            else:
                self.linear_projections.append(None)
        self.layers.append(GATConv(back_architecture[-1], num_features, heads=2, dropout=0.3, concat=False))
        self.linear_projections.append(torch.nn.Linear(back_architecture[-1], num_features))

        self.batch_norm = torch.nn.ModuleList()
        for out_dim in back_architecture[:-1]:  # BatchNorm for all layers except the last
           self.batch_norm.append(torch.nn.BatchNorm1d(out_dim)) 

    def forward(self, z, edge_index):
        for i, layer in enumerate(self.layers[:-1]):
            residual = z
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = layer(z, edge_index)  # Apply GCN
            #z = self.batch_norm[i](z)  # Apply BatchNorm
            z = F.relu(z)  # Apply activation function

            if self.linear_projections[i]:
                residual = self.linear_projections[i](residual)  # Match dimensions
            z = z + residual 
        
         #Last layer (without BatchNorm)
        return self.layers[-1](z, edge_index)
    
class graphAEAttention(torch.nn.Module):
    def __init__(self, num_features, architecture, back_architecture, dropout):
        super(graphAEAttention, self).__init__()
        self.encoder = EncoderAttention(num_features, architecture, dropout)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoderAttention(num_features, back_architecture, dropout)
        
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
    

class encoderGIN(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(encoderGIN, self).__init__()
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

class features_decoderGIN(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(features_decoderGIN, self).__init__()
        
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
    
class graphAEGIN(torch.nn.Module):
    def __init__(self, num_features, architecture):
        super(graphAEGIN, self).__init__()
        self.encoder = encoderGIN(num_features, architecture)
        self.edges_decoder = InnerProductDecoder()
        self.features_decoder = features_decoderGIN(num_features, architecture)
        
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
    